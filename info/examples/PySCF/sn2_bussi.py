"""Input configuration for SN2 reaction."""

# Set the default number of threads before importing libraries
from os import environ

environ.setdefault("OMP_NUM_THREADS", "1")  # Good default, but can be overridden by the user

# Standard Library
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

# Third Party Library
from pyscf.md.integrators import NVTBussi

# First Party Library
from distance_metrics import proton_transfer
from revo_pyscf import run

BREAK_PAIR = (0, 1)
MAKE_PAIR = (0, 5)
BREAK_CUTOFF = 7.56  # Bohr  (= 4.0 Å; C-Br counted broken)  [0.4 nm]
MAKE_CUTOFF = 2.83  # Bohr  (= 1.5 Å; C-F counted formed)  [0.15 nm]


@dataclass
class PySCFInput:
    #
    # System
    #
    topology_file_path: str = str(Path(__file__).resolve().parent / "sn2.pdb")
    system_name: str = "sn2"

    #
    # Simulation parameters
    #
    backend: Literal["cpu", "gpu"] = "gpu"
    n_walkers = 12
    n_cycles = 100
    segment_length = 10

    #
    # PySCF runner parameters
    #
    basis: str = "aug-cc-pVDZ"
    ecp: str | dict | None = None
    auxbasis: str | None = "aug-cc-pVDZ-jkfit"  # None automatically selects an appropriate auxbasis
    method: Literal["RHF", "UHF", "RKS", "UKS"] = "RKS"
    xc: str | None = "wb97x_v"
    charge: int = -1
    spin: int = 0
    dt: int = 21
    temperature_kelvin: float = 100.0
    density_grid_shape: tuple[int, int, int] | None = None

    #
    # PySCF integrator and any kwargs passed to it
    #
    integrator_cls = NVTBussi
    integrator_kwargs: dict = field(default_factory=lambda: {"taut": 4134.0})

    #
    # Distance metric and resampler parameters
    #
    distance_metric = proton_transfer(BREAK_PAIR, MAKE_PAIR)

    @dataclass
    class ResamplerParameters:
        merge_dist: float = 0.05
        char_dist: float = 0.1
        pmin: float = 1e-12
        pmax: float = 0.20

    # If resampler parameters is None, then no resampler is used
    resampler_parameters: ResamplerParameters | None = field(default_factory=ResamplerParameters)

    #
    # Boundary conditions
    #
    use_boundary_conditions: bool = True
    break_pairs: list[tuple[int, int]] = field(default_factory=lambda: [BREAK_PAIR])
    break_cutoffs: list[float] = field(default_factory=lambda: [BREAK_CUTOFF])
    make_pairs: list[tuple[int, int]] = field(default_factory=lambda: [MAKE_PAIR])
    make_cutoffs: list[float] = field(default_factory=lambda: [MAKE_CUTOFF])

    #
    # Misc
    #
    initialize_velocities: bool = True  # Initialize velocities from Maxwell-Boltzmann distribution (False uses zeros)
    unique_initial_velocities: bool = True  # Generate unique initial velocities for each walker
    use_density_fitting: bool = True  # Use density fitting with the auxbasis
    use_scanner_caching: bool = True  # Cache scanners from the previous cycle to speed up first step greatly
    scanner_cache_capacity: int | None = None  # The amount of scanners the cache can hold (None uses n_walkers)
    suppress_pyscf_output: bool = True  # Suppress PySCF gradient/velocity/position output

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


if __name__ == "__main__":
    CONFIG = PySCFInput()

    run(CONFIG)
