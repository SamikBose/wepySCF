"""Configuration for the PySCF wepy simulation maker."""

# Standard Library
from dataclasses import dataclass, field
from os import environ
from typing import Literal

# Third Party Library
from pyscf.md.integrators import LangevinMiddle, _Integrator

# First Party Library
from wepy.resampling.distances.distance import Distance


@dataclass
class PySCFSimMakerConfig:
    #
    # System
    #
    topology_file_path: str
    system_name: str

    #
    # Simulation parameters
    #
    backend: Literal["CPU", "GPU"] = "GPU"
    n_walkers: int = 4
    n_cycles: int = 5
    segment_length: int = 10

    #
    # PySCF runner parameters
    #
    basis: str = "sto-3g"
    ecp: str | dict | None = None
    auxbasis: str | None = "def2-universal-jkfit"  # None auto-selects an appropriate auxbasis
    method: Literal["RHF", "UHF", "RKS", "UKS"] = "RHF"
    xc: str | None = None
    population_method: Literal["mulliken", "meta-lowdin", "lowdin"] = "meta-lowdin"
    charge: int = 0
    spin: int = 0
    dt: int = 21
    temperature_kelvin: float = 300.0
    density_grid_shape: tuple[int, int, int] | None = None

    #
    # PySCF integrator and kwargs passed to it
    #
    integrator_cls: type[_Integrator] = LangevinMiddle
    integrator_kwargs: dict = field(default_factory=lambda: {"friction_coef": 1.0})

    #
    # Distance metric and resampler parameters
    #
    distance_metric: Distance = field(default_factory=Distance)

    @dataclass
    class ResamplerParameters:
        merge_dist: float = 0.025
        char_dist: float = 0.1
        pmin: float = 1e-12
        pmax: float = 0.99

    # If resampler_parameters is None, then no resampler is used (NoResampler)
    resampler_parameters: ResamplerParameters | None = field(default_factory=ResamplerParameters)

    #
    # Boundary conditions
    #
    use_boundary_conditions: bool = False
    break_pairs: list[tuple[int, int]] = field(default_factory=list)
    break_cutoffs: list[float] = field(default_factory=list)
    make_pairs: list[tuple[int, int]] = field(default_factory=list)
    make_cutoffs: list[float] = field(default_factory=list)

    #
    # Initial ensemble / initialization
    #
    initialization_mode: Literal["cartesian_mb", "normal_mode_ts"] = "cartesian_mb"
    # TODO: These used for normal mode ts?
    initialize_velocities: bool = True  # Draw velocities from Maxwell-Boltzmann (False uses zeros)
    unique_initial_velocities: bool = True  # Unique MB velocities per walker vs. one shared draw
    # TODO: Check these?
    initial_ensemble_seed: int | None = None  # Seed for normal_mode_ts sampling
    normal_modes_file_path: str | None = None  # Required when initialization_mode == "normal_mode_ts"

    #
    # Pathway / symmetry validation
    #
    require_competing_symmetry: bool = False
    max_competing_asymmetry_angstrom: float = 1.0e-3
    tracked_boundary_pairs: list[tuple[int, int]] = field(default_factory=list)

    #
    # Performance
    #
    use_density_fitting: bool = False  # Use density fitting with the auxbasis
    use_scanner_caching: bool = True  # Cache scanners between cycles to speed up first step
    scanner_cache_capacity: int | None = None  # None resolves to n_walkers in __post_init__
    suppress_pyscf_output: bool = True  # Suppress PySCF gradient/velocity/position output

    #
    # Output control
    #
    write_h5: bool = True
    write_dash: bool = True
    store_pickles: bool = True
    overwrite: bool = False

    #
    # Read-only / derived (populated in __post_init__)
    #
    _omp_threads_env_var: str = field(default_factory=str)
    _cuda_visible_devices_env_var: str = field(default_factory=str)
    _num_gpus_visible: int = field(default_factory=int)

    @property
    def _integrator_name(self) -> str:
        return getattr(self.integrator_cls, "__name__", "integrator")

    @property
    def output_directory(self) -> str:
        parts = [
            self.system_name,
            f"{self.n_walkers}W",
            f"{self.n_cycles}C",
            f"{self.segment_length}S",
            self._integrator_name,
            f"{self.temperature_kelvin}K",
        ]

        # Add friction/taut parameters from integrator_kwargs
        if self.integrator_kwargs is not None:
            if "friction_coef" in self.integrator_kwargs:
                parts.append(f"{self.integrator_kwargs['friction_coef']}fric")
            elif "taut" in self.integrator_kwargs:
                parts.append(f"{self.integrator_kwargs['taut']}taut")

        # Add merge distance parameter from resampler_parameters
        if self.resampler_parameters is not None:
            parts.append(f"{self.resampler_parameters.merge_dist}mergedist")

        return "_".join(parts)

    @property
    def filename_base(self) -> str:
        return f"{self.xc}_{self.basis}"

    def get_h5_path(self, output_directory: str) -> str:
        """Return the h5 path (evaluated at runtime)."""
        return f"{output_directory}/{self.filename_base}.wepy.h5"

    def get_dash_path(self, output_directory: str) -> str:
        """Return the dash path (evaluated at runtime)."""
        return f"{output_directory}/{self.filename_base}.dash.org"

    def __post_init__(self) -> None:
        if self.integrator_cls is None:
            raise ValueError("integrator_cls must be specified")

        if self.distance_metric is None:
            raise ValueError("distance_metric must be specified")

        if self.scanner_cache_capacity is None:
            self.scanner_cache_capacity = self.n_walkers

        if self.initialization_mode == "normal_mode_ts":
            if self.n_walkers % 2:
                raise ValueError("A C2-paired normal-mode ensemble requires an even walker count")
            if not self.normal_modes_file_path:
                raise ValueError("normal_modes_file_path is required for normal_mode_ts initialization")

        if self.initialization_mode not in ("cartesian_mb", "normal_mode_ts"):
            raise ValueError(f"Unknown initialization_mode: {self.initialization_mode}")

        # Environment-derived fields
        self._omp_threads_env_var = environ.get("OMP_NUM_THREADS", "")
        self._cuda_visible_devices_env_var = environ.get("CUDA_VISIBLE_DEVICES", "")
        self._num_gpus_visible = len([x for x in self._cuda_visible_devices_env_var.split(",") if x.strip()])
