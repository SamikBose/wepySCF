"""PySCF molecular simulation runner and accessory classes."""

# Standard Library
import logging

logger = logging.getLogger(__name__)
# Standard Library
import os
import uuid
from collections import OrderedDict
from time import perf_counter
from typing import Literal

# Third Party Library
import numpy as np

# TODO: Lazy imports
import pyscf.dft as pyscf_dft
import pyscf.dft.numint as pyscf_numint
import pyscf.md.integrators as pyscf_md_integrators
import pyscf.scf as pyscf_scf

# First Party Library
from wepy.runners.runner import Runner
from wepy.walker import Walker, WalkerState
from wepy.work_mapper.worker import Worker, WorkerMapper

# Supported PySCF methods
SUPPORTED_METHODS = ("RHF", "UHF", "RKS", "UKS")

# Names of the fields of PySCFState
KEYS = (
    "walker_id",
    "mol",
    "positions",
    "velocities",
    "accelerations",
    "temperature",
    "total_energy",
    "potential",
    "kinetic",
    "mo_energy",
    "charges",
    "density_matrix",
    "density_grid",
    "density_grid_origin",
    "density_grid_spacing",
    "extra_data",
)
REQUIRED_KEYS = (
    "mol",
    "positions",
    "velocities",
    "temperature",
)


# Unit metadata for reporters
UNIT_NAMES = (
    ("positions_unit", "bohr"),
    ("velocities_unit", "bohr/au"),
    ("accelerations_unit", "bohr/au^2"),
    ("temperature_unit", "kelvin"),
    ("energy_unit", "hartree"),
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
    "NVTBerendson": ("T", "taut"),
    "NVTBussi": ("T", "taut"),
    "Langevin": ("T",),
    "LangevinMiddle": ("T",),
}
TEMPERATURE_AWARE_INTEGRATORS: set[str] = {"NVTBerendson", "NVTBussi", "Langevin", "LangevinMiddle"}
RANDOM_NOISE_INTEGRATORS: set[str] = {"RandomNoiseVelocityVerlet", "NVTBussi", "Langevin", "LangevinMiddle"}


class LRUDict(OrderedDict):
    """Simple LRU cache used for the scanner cache."""

    def __init__(self, max_len=8):
        self.max_len = max_len

    def __setitem__(self, key, value):
        if key in self:
            self.move_to_end(key)
        super().__setitem__(key, value)
        if len(self) > self.max_len:
            self.popitem(last=False)


def _to_numpy(arr) -> np.ndarray:
    """Convert an array-like object to a NumPy array of floats."""
    if hasattr(arr, "get"):
        arr = arr.get()
    return np.asarray(arr, dtype=float)


def _to_cpu(obj):
    """Ensure object is a CPU object, not a GPU object."""
    if hasattr(obj, "to_cpu"):
        return obj.to_cpu()
    return obj


class PySCFState(WalkerState):
    """State of a PySCF walker, storing molecular state and walker properties."""

    KEYS = frozenset(KEYS)
    REQUIRED_KEYS = frozenset(REQUIRED_KEYS)

    def __init__(self, **kwargs):
        kwargs_set = set(kwargs)

        missing = self.REQUIRED_KEYS - kwargs_set
        if missing:
            raise ValueError(f"Missing required key(s) for PySCFState: {sorted(missing)}")
        extra = kwargs_set - set(self.KEYS)
        if extra:
            raise ValueError(f"Unexpected key(s) for PySCFState: {sorted(extra)}")

        # Store scalars as 1D feature arrays so the HDF5 reporter can extend them

        # Ensure temperature is an array
        if not isinstance(kwargs["temperature"], np.ndarray):
            kwargs["temperature"] = np.array([kwargs["temperature"]], dtype=float)

        # Fill in defaults for optional keys not provided
        defaults = {
            "walker_id": str(uuid.uuid4()),
            "accelerations": None,
            "total_energy": np.array([np.nan], dtype=float),
            "potential": np.array([np.nan], dtype=float),
            "kinetic": np.array([np.nan], dtype=float),
            "mo_energy": np.array([np.nan], dtype=float),
            "charges": np.array([np.nan], dtype=float),
        }
        if "density_grid" in kwargs:  # Add density related keys to kwargs if density grid supplied
            defaults.update(
                {
                    "density_matrix": None,
                    "density_grid_origin": np.zeros(3),
                    "density_grid_spacing": np.ones(3),
                },
            )
        for key, default_value in defaults.items():
            kwargs.setdefault(key, default_value)

        super().__init__(**kwargs)

    def get(self, key, default=None):
        return self._data.get(key, default)


class PySCFWalker(Walker):
    """Simple Walker wrapper that ensures the state is a PySCFState."""

    def __init__(self, state: PySCFState, weight):
        assert isinstance(state, PySCFState), f"state must be an instance of PySCFState not {type(state)}"
        super().__init__(state, weight)


class PySCFRunner(Runner):
    """Runner for PySCF walkers, handling state initialization and method execution."""

    SUPPORTED_METHODS = SUPPORTED_METHODS

    # TODO: Make doc with descriptions and units
    def __init__(
        self,
        backend: str = "CPU",
        method: Literal["RHF", "UHF", "RKS", "UKS"] = "RHF",
        xc: str | None = None,
        population_method: Literal["mulliken", "meta-lowdin", "lowdin"] = "meta-lowdin",
        dt: int = 21,
        integrator_cls: pyscf_md_integrators._Integrator = pyscf_md_integrators.VelocityVerlet,
        integrator_kwargs: dict | None = None,
        integrator_temperature_kelvin: float = 300.0,
        density_grid_shape: tuple[int, int, int] | None = None,
        density_grid_padding: float = 2.0,
        use_density_fitting: bool = False,
        auxbasis: str | None = None,
        use_scanner_caching: bool = False,
        scanner_cache_capacity: int = 8,
    ) -> None:
        self.backend = backend.upper()
        self.method = method.upper()
        self.auxbasis = auxbasis
        self.xc = xc
        self.population_method = population_method
        self.dt = dt
        self.integrator_cls = integrator_cls
        self.integrator_kwargs = {} if integrator_kwargs is None else dict(integrator_kwargs)
        self._integrator_temperature_kelvin = float(integrator_temperature_kelvin)
        self.density_grid_shape = density_grid_shape
        self.density_grid_padding = float(density_grid_padding)
        self._use_density_fitting = bool(use_density_fitting)
        self._use_scanner_caching = bool(use_scanner_caching)
        self.scanner_cache_capacity = scanner_cache_capacity

        if self.method not in self.SUPPORTED_METHODS:
            raise ValueError(
                f"Unsupported PySCF mean-field method '{self.method}'. "
                f"PySCF integrators only support: {self.SUPPORTED_METHODS}",
            )

        self._last_cycle_segments_split_times: list[dict] = []

    def _build_mean_field(self, mol, state):
        """Build the mean-field object for the given molecule and state."""
        if self.method == "RHF":
            mf = pyscf_scf.RHF(mol)
        elif self.method == "UHF":
            mf = pyscf_scf.UHF(mol)
        elif self.method == "RKS":
            mf = pyscf_dft.RKS(mol)
            xc = state.get("xc", self.xc)
            if xc is None:
                raise ValueError("RKS method requires an xc functional.")
            mf.xc = xc
        elif self.method == "UKS":
            mf = pyscf_dft.UKS(mol)
            xc = state.get("xc", self.xc)
            if xc is None:
                raise ValueError("UKS method requires an xc functional.")
            mf.xc = xc
        else:
            raise ValueError(f"Unsupported PySCF mean-field method '{self.method}'.")

        if self._use_density_fitting:
            mf = mf.density_fit(auxbasis=self.auxbasis)

        return mf

    def _configure_hardware(self, mf, backend: str):
        """Configure the mean-field object for the given backend."""
        if backend == "GPU":
            if hasattr(mf, "to_gpu"):
                try:
                    mf = mf.to_gpu()
                except ModuleNotFoundError as exc:
                    if getattr(exc, "name", None) == "cupy":
                        raise RuntimeError(
                            "GPU backend requested but CuPy is not installed. "
                            "Install a CuPy build compatible with your CUDA version "
                            "(e.g. cupy-cuda12x) or run with CPU backend.",
                        ) from exc
                    raise
                except AttributeError as exc:
                    raise RuntimeError(
                        "Requested GPU backend but PySCF mean-field object does not support to_gpu().",
                    ) from exc
            else:
                raise RuntimeError("Requested GPU backend but PySCF mean-field object does not support to_gpu().")

        return mf

    def _build_scanner(self, mol, state: PySCFState, backend: str):
        """Build scanner for the given molecule, state, and requested backend."""
        mf = self._build_mean_field(mol, state)
        mf = self._configure_hardware(mf, backend=backend)
        grad_method = mf.nuc_grad_method()
        if not hasattr(grad_method, "as_scanner"):
            return None

        return grad_method.as_scanner()

    def _generate_integrator_kwargs(self, integrator_cls, integrator_kwargs: dict):
        """Generate integrator kwargs from runner settings.

        Sets integrator `T` equal to `temperature_kelvin`
        for temperature-aware integrators.

        Also sets `rng` for integrators with random noise.
        """
        name = getattr(integrator_cls, "__name__", "")
        if name in TEMPERATURE_AWARE_INTEGRATORS:
            if "T" in integrator_kwargs and float(integrator_kwargs["T"]) != self._integrator_temperature_kelvin:
                logger.warning(
                    "Overriding integrator_kwargs['T']=%s to match temperature_kelvin=%s for %s",
                    integrator_kwargs["T"],
                    self._integrator_temperature_kelvin,
                    name,
                )
            integrator_kwargs["T"] = self._integrator_temperature_kelvin
        if name in RANDOM_NOISE_INTEGRATORS:
            integrator_kwargs["rng"] = np.random.Generator(np.random.PCG64(None))

        return integrator_kwargs

    def _validate_integrator_kwargs(
        self,
        integrator_cls: pyscf_md_integrators._Integrator,
        integrator_kwargs: dict,
    ):
        """Simple kwargs validation for PySCF MD integrators."""
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
        integrator_kwargs = self._generate_integrator_kwargs(self.integrator_cls, integrator_kwargs)
        self._validate_integrator_kwargs(self.integrator_cls, integrator_kwargs)

        kwargs = {"dt": self.dt, **integrator_kwargs}
        return self.integrator_cls(scanner, **kwargs)

    def _restore_integrator_values(
        self, integrator: pyscf_md_integrators._Integrator, velocities, mid_velocities, accelerations
    ):
        """Restore velocities, mid velocities (if needed), and accelerations for an integrator."""
        if velocities is not None:
            integrator.veloc = velocities
        if hasattr(integrator, "mid_veloc") and mid_velocities is not None:
            integrator.mid_veloc = mid_velocities
        if hasattr(integrator, "accel") and accelerations is not None:
            integrator.accel = accelerations

    @staticmethod
    def _lowdin_pop(mol, dm_total, s):
        """Symmetric (Lowdin) population analysis: pop_AO = diag(S^1/2 P S^1/2), aggregated per atom.

        No reference basis needed. Returns (pop_ao, charges).
        """
        e, u = np.linalg.eigh(s)
        s_half = (u * np.sqrt(e)) @ u.T  # S^{1/2}
        pop_ao = np.einsum("ij,ji->i", s_half @ dm_total, s_half)
        charges = np.zeros(mol.natm)
        aoslices = mol.aoslice_by_atom()
        for a in range(mol.natm):
            p0, p1 = aoslices[a][2], aoslices[a][3]
            charges[a] = mol.atom_charge(a) - pop_ao[p0:p1].sum()
        return pop_ao, charges

    def _population_analysis(self, mol, dm_total, s):
        """Per-atom partial charges via the configured population scheme.

        Returns (pop, charges), matching pyscf.scf.hf.mulliken_pop.
        """
        if self.population_method == "mulliken":
            return pyscf_scf.hf.mulliken_pop(mol, dm_total, s, verbose=0)
        if self.population_method == "meta-lowdin":
            # meta-Lowdin: robust, weakly basis-dependent (needs an ANO reference,
            # available for all common main-group elements incl. C/H/F/Cl)
            return pyscf_scf.hf.mulliken_meta(mol, dm_total, s=s, verbose=0)
            # return pyscf_scf.hf.mulliken_meta(mol, dm_total, verbose=0)
        if self.population_method == "lowdin":
            return self._lowdin_pop(mol, dm_total, s)  # TODO: Convert inputs to numpy?
        raise ValueError(f"Unknown population_method '{self.population_method}'")

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

    def generate_state(
        self,
        state_data,
        positions,
        integrator,
        mo_energy,
        charges,
        density_kwargs: dict,
        extra_data: dict | None = None,
    ):
        """Generate a PySCF state from the given data."""
        # Store scalar observables as 1D feature arrays (shape (1,)) so the HDF5
        # reporter can wrap them into (n_frames, 1) feature vectors
        temperature_fv = np.asarray(integrator.temperature(), dtype=float).reshape(-1)
        total_energy_fv = np.asarray(integrator.epot + integrator.ekin, dtype=float).reshape(-1)
        potential_fv = np.asarray(integrator.epot, dtype=float).reshape(-1)
        kinetic_fv = np.asarray(integrator.ekin, dtype=float).reshape(-1)

        return PySCFState(
            **{
                **state_data,
                "mol": _to_cpu(integrator.mol),  # To CPU avoids pickling GPU object (also returns a new object)
                "positions": positions,
                "velocities": integrator.veloc,
                "accelerations": getattr(integrator, "accel", None),
                "temperature": temperature_fv,
                "total_energy": total_energy_fv,
                "potential": potential_fv,
                "kinetic": kinetic_fv,
                "mo_energy": _to_numpy(mo_energy),
                "charges": _to_numpy(charges),
                **density_kwargs,
                "extra_data": extra_data,
            },
        )

    def run_segment(self, walker: PySCFWalker, segment_length: int, **kwargs: dict):
        """Run a segment of the simulation for the given walker."""
        run_segment_start = perf_counter()

        state: PySCFState = walker.state

        scanner_cache: LRUDict = kwargs.get("scanner_cache")
        if len(scanner_cache) == 0:
            scanner_cache.max_len = self.scanner_cache_capacity

        last_velocities = state.get("velocities")
        last_accelerations = state.get("accelerations")

        extra_data: dict = state.get("extra_data", {})
        last_mid_velocities = extra_data.get("mid_velocities")  # Langevin Middle

        mol = state.get("mol")

        build_scanner_start = perf_counter()

        walker_id = state.get("walker_id")
        cached_scanner = scanner_cache.get(walker_id)
        if cached_scanner is None:
            logger.debug(f"[scanner] cold start for walker {walker_id}")
            scanner = self._build_scanner(mol, state, self.backend)
        else:
            logger.debug(f"[scanner] warm start for walker {walker_id}")
            scanner = cached_scanner
            scanner.mol = mol

        build_scanner_end = perf_counter()
        build_scanner_time = build_scanner_end - build_scanner_start
        logger.debug(f"Built scanner in {build_scanner_time} sec")

        # Build integrator and restore velocities/accelerations if present
        integrator = self._build_integrator(scanner)
        self._restore_integrator_values(integrator, last_velocities, last_mid_velocities, last_accelerations)

        # If reusing acceleration from previous segment, we can skip the initialization step
        total_steps: int = segment_length if last_accelerations is not None else (segment_length + 1)

        # Integrate over the total steps
        try:
            kernel_start = perf_counter()

            integrator.kernel(steps=total_steps)

            kernel_end = perf_counter()
            kernel_time = kernel_end - kernel_start
            logger.debug(f"Integrator kernel took {kernel_time} sec")
        except RuntimeError as exc:
            raise RuntimeError("Integrator kernel execution failed.") from exc

        #
        # Create new state
        #

        energy_and_charges_start = perf_counter()

        mo_energy = scanner.base.mo_energy  # TODO: _to_numpy?

        dm = _to_numpy(scanner.base.make_rdm1())
        s = _to_numpy(scanner.base.get_ovlp())
        dm_total = dm[0] + dm[1] if dm.ndim == 3 else dm

        _, charges = self._population_analysis(scanner.base.mol, dm_total, s)

        energy_and_charges_end = perf_counter()
        energy_and_charges_time = energy_and_charges_end - energy_and_charges_start

        logger.info(f"mo_energy: {mo_energy}")
        logger.info(f"charges: {charges}")
        logger.info(f"Energy and charges calculation took {energy_and_charges_time} sec")

        positions = integrator.mol.atom_coords()

        density_time_kwargs = {}
        density_kwargs = {}
        if self.density_grid_shape is not None:
            density_calc_start = perf_counter()

            density_grid, density_grid_origin, density_grid_spacing = self._compute_density_grid(
                integrator.mol,
                dm,
                positions,
            )

            density_calc_end = perf_counter()
            density_calc_time = density_calc_end - density_calc_start
            logger.debug(f"Density calculation took {density_calc_time} sec")
            density_time_kwargs.update({"density_time": density_calc_time})

            density_kwargs.update(
                {
                    "density_matrix": dm,
                    "density_grid": density_grid,
                    "density_grid_origin": density_grid_origin,
                    "density_grid_spacing": density_grid_spacing,
                },
            )

        mid_velocities = getattr(integrator, "mid_veloc", None)
        extra_data = {"mid_velocities": mid_velocities} if mid_velocities is not None else {}

        new_state = self.generate_state(
            state_data=state._data,
            positions=positions,
            integrator=integrator,
            mo_energy=mo_energy,
            charges=charges,
            density_kwargs=density_kwargs,
            extra_data=extra_data,
        )

        if self._use_scanner_caching:
            scanner_cache[new_state["walker_id"]] = scanner  # Add the new one

        logger.info(
            f"Temperature: {new_state['temperature'][0]:.3f} K, "
            f"Potential: {new_state['potential'][0]:.6f} Ha, "
            f"Kinetic: {new_state['kinetic'][0]:.6f} Ha",
        )

        new_walker = PySCFWalker(new_state, walker.weight)

        run_segment_end = perf_counter()
        run_segment_time = run_segment_end - run_segment_start
        logger.info(f"Total internal run_segment time: {run_segment_time} sec")

        segment_split_times = {
            "build_scanner_time": build_scanner_time,
            "kernel_time": kernel_time,
            "energy_and_charges_time": energy_and_charges_time,
            **density_time_kwargs,
            "run_segment_time": run_segment_time,
        }

        self._last_cycle_segments_split_times.append(segment_split_times)

        return new_walker


# TODO: Walkers are not guarenteed to run on same GPU every time so scanner cache is only used sometimes


class PySCFCPUWorker(Worker):
    NAME_TEMPLATE = "PySCFCPUWorker-{}"
    DEFAULT_NUM_THREADS = 1

    def __init__(self, *args, **kwargs):
        num_threads = kwargs.pop("num_threads", self.DEFAULT_NUM_THREADS)
        super().__init__(*args, num_threads=num_threads, **kwargs)
        self._scanner_cache: LRUDict = LRUDict()

    def run_task(self, task):
        platform_options = {"Threads": str(self.attributes["num_threads"])}
        return task(
            backend="CPU",
            platform_kwargs=platform_options,
            scanner_cache=self._scanner_cache,
        )


class PySCFGPUWorker(Worker):
    NAME_TEMPLATE = "PySCFGPUWorker-{}"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._scanner_cache = LRUDict()

    def run(self):
        device_id = self.mapper_attributes["device_ids"][self._worker_idx]
        os.environ["CUDA_VISIBLE_DEVICES"] = str(device_id)  # Ensure GPU4PySCF only uses the assigned device
        super().run()

    def run_task(self, task):
        device_id = self.mapper_attributes["device_ids"][self._worker_idx]
        platform_options = {"DeviceIndex": str(device_id)}
        return task(
            backend="GPU",
            platform_kwargs=platform_options,
            scanner_cache=self._scanner_cache,
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
