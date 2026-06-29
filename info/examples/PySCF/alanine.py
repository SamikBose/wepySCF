"""Input configuration for alanine dipeptide."""

# Standard Library
from dataclasses import dataclass, field
from os import environ
from pathlib import Path
from typing import Literal

# Third Party Library
from pyscf.md.integrators import LangevinMiddle

# First Party Library
from distance_metrics import qm_grid_density
from revo_pyscf import run


@dataclass
class PySCFInput:
    #
    # System
    #
    topology_file_path: str = str(Path(__file__).resolve().parent / "alanine_dipeptide.pdb")
    system_name: str = "alanine"

    #
    # Simulation parameters
    #
    backend: Literal["cpu", "gpu"] = "gpu"
    n_walkers = 4
    n_cycles = 5
    segment_length = 10

    #
    # PySCF runner parameters
    #
    basis: str = "sto-3g"
    method: Literal["RHF", "UHF", "RKS", "UKS"] = "RHF"
    xc: str | None = None
    charge: int = 0
    spin: int = 0
    dt: int = 21
    temperature_kelvin: float = 300.0
    density_grid_shape: tuple[int, int, int] | None = (10, 10, 10)
    use_density_fitting: bool = False
    auxbasis: str | None = "def2-universal-jkfit"

    #
    # PySCF integrator and any kwargs passed to it
    #
    integrator_cls = LangevinMiddle
    integrator_kwargs: dict = field(default_factory=lambda: {"friction_coef": 1.0})

    #
    # Distance metric and resampler parameters
    #
    distance_metric = qm_grid_density()

    @dataclass
    class ResamplerParameters:
        merge_dist: float = 0.025
        char_dist: float = 0.1
        pmin: float = 1e-12
        pmax: float = 0.99

    # If resampler parameters is None, then no resampler is used
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
    # Misc
    #
    initialize_velocities: bool = True  # Initialize velocities from Maxwell-Boltzmann distribution (False uses zeros)
    unique_initial_velocities: bool = True  # Generate unique initial velocities for each walker
    use_scanner_caching: bool = True  # Cache scanners from the previous cycle to speed up first step greatly
    scanner_cache_capacity: int | None = None  # The amount of scanners the cache can hold (None uses n_walkers)

    #
    # Output control
    #
    write_h5 = True
    write_dash = True
    store_pickles = True
    overwrite = False

    #
    # Read only stuff for naming/logging
    #
    @property
    def _integrator_name(self) -> str:
        return getattr(self.integrator_cls, "__name__", "integrator")

    _omp_threads_env_var: str = environ.get("OMP_NUM_THREADS", "")
    _cuda_visible_devices_env_var: str = environ.get("CUDA_VISIBLE_DEVICES", "")
    _num_gpus_visible = len([x for x in _cuda_visible_devices_env_var.split(",") if x.strip()])

    @property
    def output_directory(self) -> str:
        return f"{self.system_name}_{self.n_walkers}W_{self.n_cycles}C_{self.segment_length}S_{self._integrator_name}"

    @property
    def filename_base(self) -> str:
        return f"{self.backend}_{self._omp_threads_env_var}T_{self._num_gpus_visible}G"

    def get_h5_path(self, output_directory: str) -> str:
        """Return the h5 path (evaluated at runtime)."""
        return f"{output_directory}/{self.filename_base}.wepy.h5"

    def get_dash_path(self, output_directory: str) -> str:
        """Return the dash path (evaluated at runtime)."""
        return f"{output_directory}/{self.filename_base}.dash.org"

    def __post_init__(self):
        if self.integrator_cls is None:
            raise ValueError("integrator_cls must be specified")

        if self.distance_metric is None:
            raise ValueError("distance_metric must be specified")

        if self.scanner_cache_capacity is None:
            self.scanner_cache_capacity = self.n_walkers


if __name__ == "__main__":
    CONFIG = PySCFInput()

    run(CONFIG)
