"""PySCF molecular simulation runner and accessory classes."""

# Standard Library
import logging
import os

# Third Party Library
import numpy as np

# TODO: Lazy imports
import pyscf.cc as pyscf_cc
import pyscf.dft as pyscf_dft
import pyscf.dft.numint as pyscf_numint
import pyscf.gto as pyscf_gto
import pyscf.md as pyscf_md
import pyscf.mp as pyscf_mp
import pyscf.scf as pyscf_scf

# First Party Library
from wepy.runners.runner import Runner
from wepy.walker import Walker, WalkerState
from wepy.work_mapper.task_mapper import TaskMapper, WalkerTaskProcess
from wepy.work_mapper.worker import Worker, WorkerMapper

logger = logging.getLogger(__name__)

# TODO: Enforce this
# KEYS = (
#     "symbols",
#     "positions",
#     "energy",
#     "gradients",
#     "velocities",
#     "density_matrix",
#     "density_grid",
#     "density_grid_origin",
#     "density_grid_spacing",
#     "charge",
#     "spin",
#     "basis",
#     "method",
#     "xc",
# )

# Unit metadata for reporters
UNIT_NAMES = (
    ("positions_unit", "bohr"),
    ("velocities_unit", "bohr/au"),
    ("accelerations_unit", "bohr/au^2"),
    ("potential_unit", "hartree"),
    ("kinetic_unit", "hartree"),
    ("density_grid_unit", "electron/bohr^3"),
    ("density_grid_origin_unit", "bohr"),
    ("density_grid_spacing_unit", "bohr"),
)


REQUIRED_KWARGS_BY_INTEGRATOR: dict[str, tuple] = {
    # class_name: (required keyword arguments)
    "VelocityVerlet": (),
    "RandomNoiseVelocityVerlet": (),
    "Langevin": ("T",),
    "LangevinMiddle": ("T",),
    "NVTBerendson": ("T", "taut"),
}

# Integrators whose temperature kwarg should be kept in sync with `temperature_kelvin`
_TEMPERATURE_AWARE_INTEGRATORS: set[str] = {"Langevin", "LangevinMiddle", "NVTBerendson"}


# TODO: Remove unneeded to_numpys (avoid converting CPU to GPU if not needed)
def to_numpy(x) -> np.ndarray:
    """Convert an array-like object to a NumPy array of floats.

    Fixes issue with GPU PySCF since we need to convert CuPy arrays to NumPy arrays.
    """
    if hasattr(x, "get"):
        x = x.get()
    return np.asarray(x, dtype=float)


class PySCFState(WalkerState):
    # KEYS = KEYS

    # TODO: This is the same thing as WalkerState, just with `get` added (can just remove other functions)
    # TODO: Why not just take in parameters instead of kwargs?
    # TODO: Why are we wrapping a dictionary (for deepcopy? is that even needed?)
    # def __init__(self, **kwargs):
    #     self._data = kwargs

    # def __getitem__(self, key):
    #     return self._data[key]

    def get(self, key, default=None):
        return self._data.get(key, default)

    # def dict(self):
    #     return deepcopy(self._data)


class PySCFWalker(Walker):
    def __init__(self, state, weight):
        assert isinstance(state, PySCFState), f"state must be an instance of PySCFState not {type(state)}"
        super().__init__(state, weight)


class PySCFRunner(Runner):
    SUPPORTED_METHODS = ("RHF", "UHF", "RKS", "UKS")
    # SUPPORTED_METHODS = ("RHF", "UHF", "RKS", "UKS", "MP2", "DFMP2", "CCSD") #  Others don't support scanner?

    # TODO: Make doc with descriptions and units
    # TODO: Type hints?
    def __init__(
        self,
        basis: str = "6-31g*",
        method: str = "RHF",
        xc=None,
        charge=0,
        spin=0,
        dt: float = 21.0,
        temperature_kelvin: float = 300.0,
        integrator_cls=pyscf_md.integrators.VelocityVerlet,
        integrator_kwargs: dict | None = None,
        random_seed: int | None = None,
        backend: str = "cpu",
        density_grid_shape: tuple[int, int, int] = (10, 10, 10),
        density_grid_padding: float = 2.0,
    ):
        self.basis = basis
        self.method = method.upper()
        self.xc = xc
        self.charge = charge
        self.spin = spin
        self.dt = float(dt)
        self.integrator_cls = integrator_cls
        self.integrator_kwargs = {} if integrator_kwargs is None else dict(integrator_kwargs)
        self.temperature_kelvin = float(temperature_kelvin)
        self.backend = backend.lower()
        self.density_grid_shape = tuple(density_grid_shape)
        self.density_grid_padding = float(density_grid_padding)
        pyscf_md.set_seed(random_seed)  # TODO: Walkers generating same stuff since same seed

        if self.method not in self.SUPPORTED_METHODS:
            raise ValueError(
                f"Unsupported PySCF mean-field method '{self.method}'. Must be one of: {self.SUPPORTED_METHODS}"
            )

        self._cycle_backend = None
        self._cycle_platform_kwargs = None

        self._last_cycle_segments_split_times = []  # TODO: Not used currently

    def pre_cycle(self, backend=None, platform_kwargs=None, **kwargs):
        self._cycle_backend = backend
        self._cycle_platform_kwargs = platform_kwargs

    def post_cycle(self, **kwargs):
        self._cycle_backend = None
        self._cycle_platform_kwargs = None

    def _build_molecule(self, state: PySCFState):
        symbols = state["symbols"]
        positions = state["positions"]
        atom = [(symbol, tuple(coord)) for symbol, coord in zip(symbols, positions, strict=True)]

        return pyscf_gto.M(
            atom=atom,
            basis=state.get("basis", self.basis),
            charge=state.get("charge", self.charge),
            spin=state.get("spin", self.spin),
            unit="Bohr",
        )

    def _build_mean_field(self, mol, state):
        method = state.get("method", self.method).upper()

        if method == "RHF":
            mf = pyscf_scf.RHF(mol)
        elif method == "UHF":
            mf = pyscf_scf.UHF(mol)
        elif method == "RKS":
            mf = pyscf_dft.RKS(mol)
            xc = state.get("xc", self.xc)
            if xc is None:
                raise ValueError("RKS method requires an xc functional.")
            mf.xc = xc
        elif method == "UKS":
            mf = pyscf_dft.UKS(mol)
            xc = state.get("xc", self.xc)
            if xc is None:
                raise ValueError("UKS method requires an xc functional.")
            mf.xc = xc
        else:
            raise ValueError(f"Unsupported PySCF mean-field method '{method}'.")

        return mf

    def _build_reference_mean_field(self, mol, state):
        ref_method = state.get("reference_method", None)
        if ref_method is None:
            ref_method = "UHF" if state.get("spin", self.spin) else "RHF"

        ref_state = PySCFState(**{**state._data, "method": ref_method})
        return self._build_mean_field(mol, ref_state)

    # TODO: Use this?
    def _run_quantum_step(self, mol, state, backend, platform_kwargs):
        method = state.get("method", self.method).upper()

        if method in ("RHF", "UHF", "RKS", "UKS"):
            mf = self._build_mean_field(mol, state)
            mf = self._configure_hardware(mf, backend=backend, platform_kwargs=platform_kwargs)
            energy = mf.kernel()
            gradients = to_numpy(mf.nuc_grad_method().kernel())
            density_matrix = to_numpy(mf.make_rdm1())
            return energy, gradients, density_matrix

        mf = self._build_reference_mean_field(mol, state)
        mf = self._configure_hardware(mf, backend=backend, platform_kwargs=platform_kwargs)
        mf.kernel()

        if method in ("MP2", "DFMP2"):
            post_hf = pyscf_mp.MP2(mf)
            if method == "DFMP2":
                if not hasattr(post_hf, "density_fit"):
                    raise ValueError("DFMP2 requested but MP2 object has no density_fit().")
                post_hf = post_hf.density_fit()

            post_hf.kernel()
        elif method == "CCSD":
            post_hf = pyscf_cc.CCSD(mf)
            post_hf.kernel()
        else:
            raise ValueError(f"Unsupported PySCF method '{method}'.")

        energy = getattr(post_hf, "e_tot", None)
        if energy is None:
            energy = getattr(mf, "e_tot", None)

        gradients = post_hf.nuc_grad_method().kernel()
        if hasattr(post_hf, "make_rdm1"):  # noqa: SIM108
            density_matrix = to_numpy(post_hf.make_rdm1())
        else:
            density_matrix = to_numpy(mf.make_rdm1())

        return energy, gradients, density_matrix

    def _configure_hardware(self, mf, backend: str, platform_kwargs: dict):
        if backend and str(backend).lower() == "gpu":
            device_id = platform_kwargs.get("DeviceIndex")
            if device_id is not None:
                os.environ["CUDA_VISIBLE_DEVICES"] = str(device_id)

            if hasattr(mf, "to_gpu"):
                try:
                    mf = mf.to_gpu()
                except ModuleNotFoundError as exc:
                    if getattr(exc, "name", None) == "cupy":
                        raise RuntimeError(
                            "GPU backend requested but CuPy is not installed. "
                            "Install a CuPy build compatible with your CUDA version "
                            "(e.g. cupy-cuda12x) or run with CPU backend."
                        ) from exc
                    raise
                except AttributeError as exc:
                    raise RuntimeError(
                        "Requested GPU backend but PySCF mean-field object does not support to_gpu()."
                    ) from exc
            else:
                raise RuntimeError("Requested GPU backend but PySCF mean-field object does not support to_gpu().")

        return mf

    def _method_supports_scanner(self, method):
        return method in ("RHF", "UHF", "RKS", "UKS")

    def _build_scanner(self, mol, state: PySCFState, backend: str, platform_kwargs: dict):
        """Build scanner for the requested backend."""
        mf = self._build_mean_field(mol, state)
        mf = self._configure_hardware(mf, backend=backend, platform_kwargs=platform_kwargs)
        grad_method = mf.nuc_grad_method()
        if not hasattr(grad_method, "as_scanner"):
            return None

        return grad_method.as_scanner()

    def _autoset_integrator_kwargs(self, integrator_cls, integrator_kwargs: dict):
        """Auto-set integrator kwargs derived from runner settings.

        Currently only auto-sets integrator `T` equal to `temperature_kelvin`
        for temperature-aware integrators.
        """
        name = getattr(integrator_cls, "__name__", "")
        if name in _TEMPERATURE_AWARE_INTEGRATORS:
            if "T" in integrator_kwargs and float(integrator_kwargs["T"]) != self.temperature_kelvin:
                logger.warning(
                    "Overriding integrator_kwargs['T']=%s to match temperature_kelvin=%s for %s",
                    integrator_kwargs["T"],
                    self.temperature_kelvin,
                    name,
                )
            integrator_kwargs["T"] = self.temperature_kelvin

        return integrator_kwargs

    def _validate_integrator_kwargs(self, integrator_cls, integrator_kwargs: dict):
        """Simple kwargs validation for PySCF MD integrators.

        If an integrator is unknown here, we do not validate and let PySCF
        raise a `TypeError` naturally.
        """
        name = getattr(integrator_cls, "__name__", None)
        required = REQUIRED_KWARGS_BY_INTEGRATOR.get(name)
        if name is None or required is None:
            raise ValueError(f"{name} integrator not supported.")

        missing = [arg for arg in required if arg not in integrator_kwargs]
        if missing:
            raise ValueError(f"Missing required integrator_kwargs for pyscf.md.integrators.{name}: {missing}")

    def _build_integrator(self, scanner):
        """Construct the configured PySCF MD integrator for a given scanner."""
        integrator_kwargs = {} if self.integrator_kwargs is None else self.integrator_kwargs
        integrator_kwargs = self._autoset_integrator_kwargs(self.integrator_cls, integrator_kwargs)
        self._validate_integrator_kwargs(self.integrator_cls, integrator_kwargs)

        kwargs = {"dt": self.dt, **integrator_kwargs}
        return self.integrator_cls(scanner, **kwargs)

    def _restore_integrator_values(self, integrator, velocities, mid_velocities, accelerations):
        """Restore velocities, mid velocities (if needed), and accelerations for an integrator."""
        if velocities is not None:
            integrator.veloc = velocities
        if hasattr(integrator, "mid_veloc") and mid_velocities is not None:
            integrator.mid_veloc = mid_velocities
        if hasattr(integrator, "accel") and accelerations is not None:
            integrator.accel = accelerations

    def _make_density_grid_coords(self, positions):
        mins = np.min(positions, axis=0) - self.density_grid_padding
        maxs = np.max(positions, axis=0) + self.density_grid_padding

        axes = [np.linspace(mins[i], maxs[i], self.density_grid_shape[i]) for i in range(3)]
        mesh = np.meshgrid(*axes, indexing="ij")
        coords = np.stack(mesh, axis=-1).reshape(-1, 3)

        spacing = np.array([axes[i][1] - axes[i][0] if len(axes[i]) > 1 else 1.0 for i in range(3)])

        return coords, mins, spacing

    def _compute_density_grid(self, mol, density_matrix, positions):
        if density_matrix.ndim == 3:
            density_matrix = density_matrix[0] + density_matrix[1]

        grid_coords, origin, spacing = self._make_density_grid_coords(positions)

        ao_values = pyscf_numint.eval_ao(mol, grid_coords)
        rho = pyscf_numint.eval_rho(mol, ao_values, density_matrix)
        rho_grid = rho.reshape(self.density_grid_shape)

        return rho_grid, origin, spacing

    # TODO: Check this
    def _density_matrix_from_scanner(self, scanner, mol, state: PySCFState, backend: str, platform_kwargs: dict):
        """Extract an AO-basis 1-RDM from a PySCF scanner.

        This is intended to mirror the old per-step behavior where we ran
        `energy, gradients = scanner(mol)` and then read back
        `scanner.base.make_rdm1()`.

        Strategy:
        1) If the scanner (or its `.base`) exposes `make_rdm1()`, use it.
        2) Otherwise, rerun an SCF calculation at the provided `mol` geometry.

        Returns:

        dm : np.ndarray
            AO density matrix. For UHF/UKS this may be shape (2,nao,nao).
        """
        if scanner is not None and hasattr(scanner, "make_rdm1"):
            try:
                return to_numpy(scanner.make_rdm1())
            except Exception as exc:
                logger.debug("Failed to get density matrix from scanner.make_rdm1(): %s", exc)

        # Common PySCF pattern: grad_scanner.base is a SCF single-point scanner
        # (see pyscf.scf.hf.as_scanner), which generally supports make_rdm1().
        scan_base = getattr(scanner, "base", None)
        if scan_base is not None and hasattr(scan_base, "make_rdm1"):
            try:
                return to_numpy(scan_base.make_rdm1())
            except Exception as exc:
                logger.debug("Failed to get density matrix from scanner.base.make_rdm1(): %s", exc)

        # Some scanner wrappers may stash the mean-field object under `.base` or `._scf`.
        for obj in (
            getattr(scan_base, "base", None),
            getattr(scan_base, "_scf", None),
            getattr(scanner, "_scf", None),
        ):
            if obj is None or not hasattr(obj, "make_rdm1"):
                continue
            try:
                return to_numpy(obj.make_rdm1())
            except Exception as exc:
                logger.debug("Failed to get density matrix from %s.make_rdm1(): %s", type(obj), exc)

        # Fall back: rerun an SCF calculation at the final geometry. This is more
        # expensive but is robust.
        mf = self._build_mean_field(mol, state)
        mf = self._configure_hardware(mf, backend=backend, platform_kwargs=platform_kwargs)
        mf.kernel()
        return to_numpy(mf.make_rdm1())

    def generate_state(
        self,
        state_data,
        positions,
        velocities,
        accelerations,
        potential,
        kinetic,
        density_matrix,
        density_grid,
        density_grid_origin,
        density_grid_spacing,
        extra_data: dict = None,
    ):
        # Store scalar observables as 1D feature arrays (shape (1,)) so the HDF5
        # reporter can wrap them into (n_frames, 1) feature vectors.
        potential_fv = np.asarray(potential, dtype=float).reshape(-1)
        kinetic_fv = np.asarray(kinetic, dtype=float).reshape(-1)

        return PySCFState(
            **{
                **state_data,
                "positions": positions,
                "velocities": velocities,
                "accelerations": accelerations,
                "potential": potential_fv,
                "kinetic": kinetic_fv,
                "density_matrix": density_matrix,
                "density_grid": density_grid,
                "density_grid_origin": density_grid_origin,
                "density_grid_spacing": density_grid_spacing,
                "extra_data": extra_data,
            }
        )

    def run_segment(self, walker: PySCFWalker, segment_length: int, **kwargs: dict):
        state: PySCFState = walker.state

        backend = kwargs.get("backend", self._cycle_backend or self.backend)
        platform_kwargs: dict = kwargs.get("platform_kwargs") or self._cycle_platform_kwargs or {}

        positions = state["positions"].copy()
        last_velocities = state.get("velocities")
        last_accelerations = state.get("accelerations")

        extra_data: dict = state.get("extra_data", {})
        last_mid_velocities = extra_data.get("mid_velocities")  # Langevin Middle

        # TODO: Can we reuse these to avoid from-scratch init?
        # last_density_matrix = state.get("density_matrix", None)
        # last_density_grid = state.get("density_grid", np.zeros(self.density_grid_shape))
        # last_density_grid_origin = state.get("density_grid_origin", np.zeros(3))
        # last_density_grid_spacing = state.get("density_grid_spacing", np.ones(3))

        scanner = None
        integrator = None

        # TODO: Need to have method in state? Will this ever change? (probably not?)
        state_method = state.get("method", self.method).upper()
        if not self._method_supports_scanner(state_method):
            raise NotImplementedError("Scanner only supported for RHF/UHF/RKS/UKS currently.")

        # Build a SCF gradient scanner once per segment (reused by the integrator)
        init_state = PySCFState(
            **{
                **state._data,  # TODO: Deep copy this? Why are we storing this? (seems like this is a nested dict of previous states)
                "positions": positions,
            }
        )
        init_mol = self._build_molecule(init_state)

        scanner = self._build_scanner(init_mol, init_state, backend, platform_kwargs)
        integrator = self._build_integrator(scanner)

        # Restore velocities and accelerations if present
        self._restore_integrator_values(integrator, last_velocities, last_mid_velocities, last_accelerations)

        # If reusing acceleration from previous segment, we can skip the initialization step
        total_steps: int = segment_length if last_accelerations is not None else (segment_length + 1)

        # Integrate over the total steps
        try:
            integrator.kernel(steps=total_steps)
        except RuntimeError as exc:
            raise RuntimeError("Integrator kernel execution failed.") from exc

        positions = integrator.mol.atom_coords()
        velocities = integrator.veloc
        accelerations = integrator.accel if hasattr(integrator, "accel") else None

        potential = integrator.epot
        kinetic = integrator.ekin

        mid_velocities = integrator.mid_veloc if hasattr(integrator, "mid_veloc") else None
        extra_data = {"mid_velocities": mid_velocities} if mid_velocities is not None else {}

        # TODO: Check this calculation
        density_matrix = self._density_matrix_from_scanner(
            scanner,
            integrator.mol,
            init_state,
            backend=backend,
            platform_kwargs=platform_kwargs,
        )
        density_grid, density_grid_origin, density_grid_spacing = self._compute_density_grid(
            integrator.mol,
            density_matrix,
            positions,
        )

        new_state = self.generate_state(
            state._data,
            positions,
            velocities,
            accelerations,
            potential,
            kinetic,
            density_matrix,
            density_grid,
            density_grid_origin,
            density_grid_spacing,
            extra_data,
        )

        if isinstance(walker, PySCFWalker):
            return PySCFWalker(new_state, walker.weight)
        return Walker(
            new_state, walker.weight
        )  # TODO: Does it even make sense to not return a PySCF walker here? Shouldn't be able to call this on non PySCF?


class PySCFCPUWorker(Worker):
    NAME_TEMPLATE = "PySCFCPUWorker-{}"
    DEFAULT_NUM_THREADS = 1

    def __init__(self, *args, **kwargs):
        num_threads = self.DEFAULT_NUM_THREADS if "num_threads" not in kwargs else kwargs.pop("num_threads")
        super().__init__(*args, num_threads=num_threads, **kwargs)

    def run_task(self, task):
        platform_options = {"Threads": str(self.attributes["num_threads"])}
        return task(backend="cpu", platform_kwargs=platform_options)


class PySCFGPUWorker(Worker):
    NAME_TEMPLATE = "PySCFGPUWorker-{}"

    def run_task(self, task):
        device_id = self.mapper_attributes["device_ids"][self._worker_idx]
        platform_options = {"DeviceIndex": str(device_id)}
        return task(backend="gpu", platform_kwargs=platform_options)


class PySCFCPUWalkerTaskProcess(WalkerTaskProcess):
    NAME_TEMPLATE = "PySCF_CPU_Walker_Task-{}"

    def run_task(self, task):
        if "num_threads" in self.mapper_attributes:
            num_threads = self.mapper_attributes["num_threads"]
            platform_options = {"Threads": str(num_threads)}
        else:
            platform_options = {}

        return task(backend="cpu", platform_kwargs=platform_options)


class PySCFGPUWalkerTaskProcess(WalkerTaskProcess):
    NAME_TEMPLATE = "PySCF_GPU_Walker_Task-{}"

    def run_task(self, task):
        device_id = self.mapper_attributes["device_ids"][self._worker_idx]
        platform_options = {"DeviceIndex": str(device_id)}
        return task(backend="gpu", platform_kwargs=platform_options)


# TODO: Remove these?
class PySCFCPUTaskMapper(TaskMapper):
    """Convenience TaskMapper for CPU walker-level parallelism."""

    def __init__(self, num_workers=None, **kwargs):
        super().__init__(
            walker_task_type=PySCFCPUWalkerTaskProcess,
            num_workers=num_workers,
            **kwargs,
        )


class PySCFGPUTaskMapper(TaskMapper):
    """Convenience TaskMapper for GPU walker-level parallelism."""

    def __init__(self, num_workers=None, platform="CUDA", device_ids=None, **kwargs):
        if device_ids is None:
            raise ValueError("device_ids must be provided for PySCFGPUTaskMapper")

        if num_workers is None:
            num_workers = len(device_ids)

        super().__init__(
            walker_task_type=PySCFGPUWalkerTaskProcess,
            num_workers=num_workers,
            platform=platform,
            device_ids=device_ids,
            **kwargs,
        )


class PySCFCPUWorkerMapper(WorkerMapper):
    """Convenience WorkerMapper for CPU walker-level parallelism."""

    def __init__(self, num_workers=None, **kwargs):
        super().__init__(
            worker_type=PySCFCPUWorker,
            num_workers=num_workers,
            **kwargs,
        )


class PySCFGPUWorkerMapper(WorkerMapper):
    """Convenience WorkerMapper for GPU walker-level parallelism."""

    def __init__(self, num_workers=None, platform="CUDA", device_ids=None, **kwargs):
        if device_ids is None:
            raise ValueError("device_ids must be provided for PySCFGPUWorkerMapper")

        if num_workers is None:
            num_workers = len(device_ids)

        super().__init__(
            worker_type=PySCFGPUWorker,
            num_workers=num_workers,
            platform=platform,
            device_ids=device_ids,
            **kwargs,
        )
