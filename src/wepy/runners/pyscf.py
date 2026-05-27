"""PySCF molecular simulation runner and accessory classes."""

# Standard Library
import logging
import os
from copy import deepcopy

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

# from pyscf.lib import param as pyscf_param
# First Party Library
from wepy.runners.runner import Runner
from wepy.walker import Walker, WalkerState
from wepy.work_mapper.task_mapper import TaskMapper, WalkerTaskProcess
from wepy.work_mapper.worker import Worker, WorkerMapper

logger = logging.getLogger(__name__)

KEYS = (
    "symbols",
    "positions",
    "energy",
    "gradients",
    "velocities",
    "density_matrix",
    "density_grid",
    "density_grid_origin",
    "density_grid_spacing",
    "charge",
    "spin",
    "basis",
    "method",
    "xc",
    "unit",
    "segment_step_idx",
)

UNIT_NAMES = (
    ("positions_unit", "angstrom"),
    ("energy_unit", "hartree"),
    ("gradients_unit", "hartree/bohr"),
    ("velocities_unit", "bohr/au"),
    ("density_grid_unit", "electron/bohr^3"),
)


REQUIRED_KWARGS_BY_INTEGRATOR = {
    # class_name: required keyword arguments
    "VelocityVerlet": (),
    "RandomNoiseVelocityVerlet": (),
    "Langevin": ("T",),
    "LangevinMiddle": ("T",),
    "NVTBerendson": ("T", "taut"),
}

# Integrators whose temperature kwarg should be kept in sync with `temperature_kelvin`
_TEMPERATURE_AWARE_INTEGRATORS = {"Langevin", "LangevinMiddle", "NVTBerendson"}


def to_numpy(x) -> np.ndarray:
    """Convert an array-like object to a NumPy array of floats.

    Fixes issue with GPU PySCF since we need to convert CuPy arrays to NumPy arrays.
    """
    if hasattr(x, "get"):
        x = x.get()
    return np.asarray(x, dtype=float)


class PySCFState(WalkerState):
    KEYS = KEYS

    def __init__(self, **kwargs):
        self._data = kwargs

    def __getitem__(self, key):
        return self._data[key]

    def get(self, key, default=None):
        return self._data.get(key, default)

    def dict(self):
        return deepcopy(self._data)


class PySCFWalker(Walker):
    def __init__(self, state, weight):
        assert isinstance(state, PySCFState), f"state must be an instance of PySCFState not {type(state)}"
        super().__init__(state, weight)


class PySCFRunner(Runner):
    SUPPORTED_METHODS = ("RHF", "UHF", "RKS", "UKS", "MP2", "DFMP2", "CCSD")

    def __init__(
        self,
        basis="6-31g*",
        method="RHF",
        xc=None,
        charge=0,
        spin=0,
        unit="Angstrom",
        step_size=1e-3,
        dt=21,  # TODO: Add to PySCFState or not since always constant? Same logic for others?
        temperature_kelvin=300.0,
        integrator_cls=pyscf_md.integrators.VelocityVerlet,
        integrator_kwargs=None,
        random_seed=None,
        backend="cpu",
        density_grid_shape=(10, 10, 10),
        density_grid_padding=2.0,  # TODO: Interpreted in Bohr right now. Use au?
        gpu_fallback_cpu_on_error=False,
    ):
        self.basis = basis
        self.method = method.upper()
        self.xc = xc
        self.charge = charge
        self.spin = spin
        self.unit = unit
        self.step_size = float(step_size)
        self.dt = dt
        self.integrator_cls = integrator_cls
        self.integrator_kwargs = {} if integrator_kwargs is None else dict(integrator_kwargs)
        self.temperature_kelvin = float(temperature_kelvin)
        self.backend = backend
        self.density_grid_shape = tuple(density_grid_shape)
        self.density_grid_padding = float(density_grid_padding)
        self.gpu_fallback_cpu_on_error = gpu_fallback_cpu_on_error
        pyscf_md.set_seed(random_seed)

        if self.method not in self.SUPPORTED_METHODS:
            raise ValueError(
                f"Unsupported PySCF mean-field method '{self.method}'. Must be one of: {self.SUPPORTED_METHODS}"
            )

        # TODO: Really needed?
        # if self.dynamics_mode not in self.SUPPORTED_DYNAMICS_MODES:
        #     raise ValueError(
        #         "Unsupported PySCF dynamics mode "
        #         f"'{self.dynamics_mode}'. Must be one of: {self.SUPPORTED_DYNAMICS_MODES}"
        #     )

        self._cycle_backend = None
        self._cycle_platform_kwargs = None

        self._last_cycle_segments_split_times = []

    def pre_cycle(self, backend=None, platform_kwargs=None, **kwargs):
        self._cycle_backend = backend
        self._cycle_platform_kwargs = platform_kwargs

    def post_cycle(self, **kwargs):
        self._cycle_backend = None
        self._cycle_platform_kwargs = None

    def _build_molecule(self, state):
        symbols = state["symbols"]
        positions = np.asarray(state["positions"], dtype=float)
        atom = [(symbol, tuple(coord)) for symbol, coord in zip(symbols, positions, strict=True)]

        return pyscf_gto.M(
            atom=atom,
            basis=state.get("basis", self.basis),
            charge=state.get("charge", self.charge),
            spin=state.get("spin", self.spin),
            unit=state.get("unit", self.unit),
        )

    # def _coords_to_bohr(self, coords, unit):
    #     """Convert coordinates to Bohr.

    #     PySCF's MD integrators propagate geometries in Bohr internally and call
    #     `mol.set_geom_(..., unit='B')`. If the molecule was built with
    #     `mol.unit = 'Angstrom'`, PySCF will warn that Mole.unit is changed to B.

    #     We avoid that warning by building the integrator molecule in Bohr.
    #     """
    #     coords = np.asarray(coords, dtype=float)
    #     if unit is None:
    #         unit = self.unit

    #     # PySCF considers strings starting with 'B' or 'AU' as atomic units.
    #     if isinstance(unit, str) and unit.upper().startswith(("B", "AU")):
    #         return coords

    #     # Default assumption: Angstrom
    #     return coords / pyscf_param.BOHR

    def _validate_integrator_kwargs(self, integrator_cls, integrator_kwargs):
        """Simple kwargs validation for PySCF MD integrators.

        If an integrator is unknown here, we do not validate and let PySCF
        raise a `TypeError` naturally.
        """
        name = getattr(integrator_cls, "__name__", None)
        required = REQUIRED_KWARGS_BY_INTEGRATOR.get(name)
        if name is None or required is None:
            return

        missing = [arg for arg in required if arg not in integrator_kwargs]
        if missing:
            raise ValueError(f"Missing required integrator_kwargs for pyscf.md.integrators.{name}: {missing}")

    def _autoset_integrator_kwargs(self, integrator_cls, integrator_kwargs):
        """Auto-set integrator kwargs derived from runner settings.

        Currently only auto-sets integrator `T` equal to `temperature_kelvin`
        for temperature-aware integrators.
        """
        name = getattr(integrator_cls, "__name__", "")
        if name in _TEMPERATURE_AWARE_INTEGRATORS:
            desired_T = self.temperature_kelvin
            if "T" in integrator_kwargs and float(integrator_kwargs["T"]) != desired_T:
                logger.warning(
                    "Overriding integrator_kwargs['T']=%s to match temperature_kelvin=%s for %s",
                    integrator_kwargs["T"],
                    desired_T,
                    name,
                )
            integrator_kwargs["T"] = desired_T

        return integrator_kwargs

    def _build_integrator(self, scanner):
        """Construct the configured PySCF MD integrator for a given scanner."""
        integrator_kwargs = {} if self.integrator_kwargs is None else dict(self.integrator_kwargs)
        integrator_kwargs = self._autoset_integrator_kwargs(self.integrator_cls, integrator_kwargs)
        self._validate_integrator_kwargs(self.integrator_cls, integrator_kwargs)

        kwargs = {"dt": self.dt, **integrator_kwargs}
        return self.integrator_cls(scanner, **kwargs)

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

    def _method_supports_scanner(self, method):
        return method in ("RHF", "UHF", "RKS", "UKS")

    # TODO: Use this?
    # def _run_quantum_step(self, mol, state, backend, platform_kwargs):
    #     method = state.get("method", self.method).upper()

    #     if method in ("RHF", "UHF", "RKS", "UKS"):
    #         mf = self._build_mean_field(mol, state)
    #         mf = self._configure_hardware(mf, backend=backend, platform_kwargs=platform_kwargs)
    #         energy = mf.kernel()
    #         gradients = to_numpy(mf.nuc_grad_method().kernel())
    #         density_matrix = to_numpy(mf.make_rdm1())
    #         return energy, gradients, density_matrix

    #     mf = self._build_reference_mean_field(mol, state)
    #     mf = self._configure_hardware(mf, backend=backend, platform_kwargs=platform_kwargs)
    #     mf.kernel()

    #     if method in ("MP2", "DFMP2"):
    #         post_hf = pyscf_mp.MP2(mf)
    #         if method == "DFMP2":
    #             if not hasattr(post_hf, "density_fit"):
    #                 raise ValueError("DFMP2 requested but MP2 object has no density_fit().")
    #             post_hf = post_hf.density_fit()

    #         post_hf.kernel()
    #     elif method == "CCSD":
    #         post_hf = pyscf_cc.CCSD(mf)
    #         post_hf.kernel()
    #     else:
    #         raise ValueError(f"Unsupported PySCF method '{method}'.")

    #     energy = getattr(post_hf, "e_tot", None)
    #     if energy is None:
    #         energy = getattr(mf, "e_tot", None)

    #     gradients = post_hf.nuc_grad_method().kernel()
    #     if hasattr(post_hf, "make_rdm1"):  # noqa: SIM108
    #         density_matrix = to_numpy(post_hf.make_rdm1())
    #     else:
    #         density_matrix = to_numpy(mf.make_rdm1())

    #     return energy, gradients, density_matrix

    def _configure_hardware(self, mf, backend="cpu", platform_kwargs=None):
        platform_kwargs = platform_kwargs or {}

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

    def _is_gpu_runtime_error(self, exc):
        msg = str(exc).lower()
        gpu_signatures = (
            "unsupported toolchain",
            "failed in block_diag kernel",
            "cuda error",
        )
        return any(sig in msg for sig in gpu_signatures)

    def _build_gradient_scanner(self, mf):
        grad_method = mf.nuc_grad_method()
        if not hasattr(grad_method, "as_scanner"):
            return None

        return grad_method.as_scanner()

    def _build_scanner_and_integrator(self, mol, state, backend, platform_kwargs):
        """Build (scanner, integrator) for the requested backend.

        This is used by MD segments for both initial construction and CPU fallback.
        """
        mf = self._build_mean_field(mol, state)
        mf = self._configure_hardware(mf, backend=backend, platform_kwargs=platform_kwargs)
        scanner = self._build_gradient_scanner(mf)
        if scanner is None:
            return None, None

        integrator = self._build_integrator(scanner)
        return scanner, integrator

    def _restore_integrator_velocities(self, integrator, velocities, mid_velocities):
        if velocities is not None:
            integrator.veloc = np.asarray(velocities, dtype=float)
        if hasattr(integrator, "mid_veloc") and mid_velocities is not None:
            integrator.mid_veloc = np.asarray(mid_velocities, dtype=float)

    def _make_density_grid_coords(self, positions):
        mins = np.min(positions, axis=0) - self.density_grid_padding
        maxs = np.max(positions, axis=0) + self.density_grid_padding

        axes = [np.linspace(mins[i], maxs[i], self.density_grid_shape[i]) for i in range(3)]
        mesh = np.meshgrid(*axes, indexing="ij")
        coords = np.stack(mesh, axis=-1).reshape(-1, 3)

        spacing = np.array([axes[i][1] - axes[i][0] if len(axes[i]) > 1 else 1.0 for i in range(3)])

        return coords, mins, spacing

    def _compute_density_grid(self, mol, density_matrix, positions_bohr):
        """Compute electron density on a regular grid around the geometry.

        PySCF AO evaluation (`pyscf.dft.numint.eval_ao`) expects grid coordinates
        in Bohr (atomic units) regardless of how the molecule geometry was
        originally specified.

        For simplicity and consistency with `density_grid_unit = electron/bohr^3`,
        this function always operates in Bohr.
        """

        dm = np.asarray(density_matrix)
        if dm.ndim == 3:
            dm = dm[0] + dm[1]

        grid_coords, origin, spacing = self._make_density_grid_coords(positions_bohr)

        ao_values = pyscf_numint.eval_ao(mol, grid_coords)
        rho = pyscf_numint.eval_rho(mol, ao_values, dm)
        rho_grid = np.asarray(rho, dtype=float).reshape(self.density_grid_shape)

        return rho_grid, origin, spacing

    # TODO: Check this
    def _density_matrix_from_scanner(self, scanner, mol, state, backend, platform_kwargs):
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
        energy,
        gradients,  # TODO: Do we need to store the gradients anymore since we can just get acceleration directly?
        velocities,
        segment_step_idx,  # TODO: Still needed since we don't process steps individually (all at once with integrator)
        density_matrix,
        density_grid,
        density_grid_origin,
        density_grid_spacing,
        extra_state_data=None,
    ):
        return PySCFState(
            **{
                **state_data,
                "positions": positions,
                "energy": np.array([np.nan if energy is None else float(np.asarray(energy).ravel()[0])]),
                "gradients": gradients,
                "velocities": velocities,
                "segment_step_idx": np.array([int(segment_step_idx)]),
                "density_matrix": density_matrix,
                "density_grid": density_grid,
                "density_grid_origin": density_grid_origin,
                "density_grid_spacing": density_grid_spacing,
                "extra_state_data": extra_state_data,
            }
        )

    def run_segment(self, walker, segment_length, **kwargs):
        state = walker.state
        positions = np.asarray(state["positions"], dtype=float).copy()

        total_steps = int(segment_length)
        if total_steps <= 0:
            raise ValueError("segment_length must be > 0")

        backend = kwargs.get("backend", self._cycle_backend or self.backend)
        platform_kwargs = kwargs.get("platform_kwargs", self._cycle_platform_kwargs or {})

        last_energy = state.get("energy", None)
        last_gradients = state.get("gradients", np.zeros_like(positions))
        last_velocities = state.get("velocities", np.zeros_like(positions))

        extra_state = state.get("extra_state_data", {})
        last_mid_velocities = state.get("mid_velocities", extra_state.get("mid_velocities", None))
        last_accel = state.get("accel", extra_state.get("accel", None))

        # last_density_matrix = np.zeros((positions.shape[0], positions.shape[0]))
        last_density_matrix = state.get("density_matrix", None)  # TODO: Use zeros(pos[0], pos[0]) or None?
        last_density_grid = state.get("density_grid", np.zeros(self.density_grid_shape))
        last_density_grid_origin = state.get("density_grid_origin", np.zeros(3))
        last_density_grid_spacing = state.get("density_grid_spacing", np.ones(3))
        segment_step_idx = 0

        scanner = None
        integrator = None
        allow_gpu_fallback = kwargs.get("gpu_fallback_cpu_on_error", self.gpu_fallback_cpu_on_error)

        state_method = state.get("method", self.method).upper()
        state_unit = state.get("unit", self.unit)

        # Build a SCF gradient scanner once per segment (reused by the integrator)
        # # NOTE: PySCF MD integrators propagate in Bohr and call mol.set_geom_(..., unit='B').
        # # To avoid unit-change warnings (Angstrom -> B) and to ensure consistent geometry,
        # # we build the integrator molecule in Bohr.
        if self._method_supports_scanner(state_method):
            # init_positions_bohr = self._coords_to_bohr(positions, state_unit)
            init_state = PySCFState(
                **{
                    **state._data,
                    "positions": positions,
                    # "positions": init_positions_bohr,
                    # "unit": "Bohr",
                    "segment_step_idx": 0,
                }
            )
            init_mol = self._build_molecule(init_state)

            try:
                scanner, integrator = self._build_scanner_and_integrator(init_mol, init_state, backend, platform_kwargs)
            except RuntimeError as exc:
                if backend == "gpu" and self._is_gpu_runtime_error(exc) and allow_gpu_fallback:
                    logger.warning("GPU initialization failed (%s); falling back to CPU for this segment.", exc)
                    backend = "cpu"
                    platform_kwargs = {}
                    scanner, integrator = self._build_scanner_and_integrator(
                        init_mol, init_state, backend, platform_kwargs
                    )
                else:
                    raise

        if scanner is None:
            raise NotImplementedError("Scanner only supported for RHF/UHF/RKS/UKS currently.")

        # Restore velocities if present
        self._restore_integrator_velocities(integrator, last_velocities, last_mid_velocities)

        # Reuse acceleration from previous segment to skip extra integrator initialization step
        seeded_accel = False
        if last_accel is not None and hasattr(integrator, "accel"):
            integrator.accel = last_accel
            seeded_accel = True

        steps_to_run = total_steps if seeded_accel else (total_steps + 1)  # +1 if not reusing to account for init step

        # Integrate over the total steps
        try:
            integrator.kernel(steps=steps_to_run)
            # integrator.kernel(steps=steps_to_run, dump_flags=False, verbose=0) # Silent
        except RuntimeError as exc:  # e.g. GPU runtime failure during execution
            if backend == "gpu" and self._is_gpu_runtime_error(exc) and allow_gpu_fallback:
                logger.warning("GPU execution failed (%s); retrying this segment on CPU.", exc)
                backend = "cpu"
                platform_kwargs = {}
                scanner, integrator = self._build_scanner_and_integrator(init_mol, init_state, backend, platform_kwargs)
                self._restore_integrator_velocities(integrator, last_velocities, last_mid_velocities)

                integrator.kernel(steps=steps_to_run)
                # integrator.kernel(steps=steps_to_run, dump_flags=False, verbose=0) # Silent
            else:
                raise

        # Collect final step data (successful run, regardless of whether CPU fallback occurred)

        # Convert positions back to the state's requested unit for storage
        positions = integrator.mol.atom_coords(unit=state_unit)

        # NOTE: energy stored here is total energy (potential + kinetic). If you want
        # old behavior (potential energy only), store integrator.epot instead.
        last_energy = integrator.epot + integrator.ekin

        if hasattr(integrator, "accel") and getattr(integrator, "_masses", None) is not None:
            last_gradients = -integrator.accel * integrator._masses.reshape(-1, 1)

        last_velocities = integrator.veloc
        if hasattr(integrator, "mid_veloc"):
            last_mid_velocities = integrator.mid_veloc

        # Density / grid at final geometry
        last_density_matrix = self._density_matrix_from_scanner(
            scanner,
            integrator.mol,
            init_state,
            backend=backend,
            platform_kwargs=platform_kwargs,
        )

        positions_bohr = integrator.mol.atom_coords(unit="Bohr")  # PySCF eval_ao expects Bohr coords
        last_density_grid, last_density_grid_origin, last_density_grid_spacing = self._compute_density_grid(
            integrator.mol,
            last_density_matrix,
            positions_bohr,
        )
        segment_step_idx = total_steps  # TODO: Remove?

        # Persist integrator state that helps us avoid redundant work next segment.
        extra_state_data = None
        if last_mid_velocities is not None or getattr(integrator, "accel", None) is not None:
            extra_state_data = {}
            if last_mid_velocities is not None:
                extra_state_data["mid_velocities"] = last_mid_velocities
            if getattr(integrator, "accel", None) is not None:
                extra_state_data["accel"] = to_numpy(integrator.accel)

        new_state = self.generate_state(
            state._data,
            positions=positions,
            energy=last_energy,
            gradients=last_gradients,
            velocities=last_velocities,
            segment_step_idx=segment_step_idx,
            density_matrix=last_density_matrix,
            density_grid=last_density_grid,
            density_grid_origin=last_density_grid_origin,
            density_grid_spacing=last_density_grid_spacing,
            extra_state_data=extra_state_data,
        )

        if isinstance(walker, PySCFWalker):
            return PySCFWalker(new_state, walker.weight)
        return Walker(new_state, walker.weight)


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
