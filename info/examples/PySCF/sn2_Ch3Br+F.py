"""Input configuration for SN2 reaction."""

# Standard Library
from dataclasses import dataclass, field
from os import environ
from pathlib import Path
from typing import Literal

# Third Party Library
from pyscf.md.integrators import LangevinMiddle

# First Party Library
from distance_metrics import proton_transfer
from revo_pyscf import run
import pyscf.md as pyscf_md

BREAK_PAIR = (0, 1)
MAKE_PAIR = (0, 5)
#BREAK_CUTOFF = 0.4  # nm
#MAKE_CUTOFF = 0.15  # nm
BREAK_CUTOFF = 7.56   # Bohr  (= 4.0 Å; C-Br counted broken)   [was 0.4, read as Bohr]
MAKE_CUTOFF  = 2.83   # Bohr  (= 1.5 Å; C-F counted formed)    [was 0.15, read as Bohr]

@dataclass
class PySCFInput:
    #
    # System
    topology_file_path: str = str("ch3br+f_opt.pdb")
    system_name: str = "CH3Br+F"


    #
    # Simulation parameters
    backend: Literal["cpu", "gpu"] = "gpu"
    n_walkers = 24
    n_cycles = 100
    segment_length = 10

    #
    # PySCF runner parameters
    basis: str | dict = field(default_factory=lambda: {"Br": "aug-cc-pvdz-pp", "default": "aug-cc-pvdz"})
    ecp:   str | dict | None = field(default_factory=lambda: {"Br": "aug-cc-pvdz-pp"})
    auxbasis: str | None = None
    #basis = {"Br": "aug-cc-pvdz-pp", "default": "aug-cc-pvdz"}
    #ecp   = {"Br": "aug-cc-pvdz-pp"}
    method: Literal["RHF", "UHF", "RKS", "UKS"] = "RKS"
    xc: str | None = "wb97x_v"
    charge: int = -1
    spin: int = 0
    dt: int = 21
    temperature_kelvin: float = 300.0
    friction_coef: float = 0.1          #real field now 
    density_grid_shape: tuple[int, int, int] | None = None
    use_density_fitting: bool = True
    #auxbasis: str | None = "aug-cc-pVDZ-jkfit"


    integrator_cls: type = pyscf_md.integrators.LangevinMiddle
    # Leave empty here; built from self.friction_coef in __post_init__ (as a float!)
    integrator_kwargs: dict = field(default_factory=dict)
    
    # Distance metric and resamplwqer parameters
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
    use_boundary_conditions: bool = True
    break_pairs: list[tuple[int, int]] = field(default_factory=lambda: [BREAK_PAIR])
    break_cutoffs: list[float] = field(default_factory=lambda: [BREAK_CUTOFF])
    make_pairs: list[tuple[int, int]] = field(default_factory=lambda: [MAKE_PAIR])
    make_cutoffs: list[float] = field(default_factory=lambda: [MAKE_CUTOFF])

    #
    # Misc
    initialize_velocities: bool = True  # Initialize velocities from Maxwell-Boltzmann distribution (False uses zeros)
    unique_initial_velocities: bool = True  # Generate unique initial velocities for each walker
    use_scanner_caching: bool = True  # Cache scanners from the previous cycle to speed up first step greatly
    scanner_cache_capacity: int | None = None  # The amount of scanners the cache can hold (None uses n_walkers)
    suppress_pyscf_output: bool = True  # Suppress PySCF gradient/velocity/position output 
    
    #
    # Output control
    write_h5 = True
    write_dash = True
    store_pickles = True
    overwrite = False

    #
    # Read only stuff for naming/logging
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
            f"{self.friction_coef}fric",
                ]
        if self.resampler_parameters is not None:
            parts.append(f"{self.resampler_parameters.merge_dist}mergedist")
        return "_".join(parts)


    @property
    def _basis_label(self) -> str:
        if isinstance(self.basis, dict):
            heavy = "".join(f"{k}-{v}" for k, v in self.basis.items() if k != "default")
            return f"{self.basis.get('default', 'mixed')}+{heavy}"
        return str(self.basis)

    @property
    def filename_base(self) -> str:
        return f"{self.xc}_{self._basis_label}"

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

        if not self.integrator_kwargs:
            self.integrator_kwargs = {"friction_coef": self.friction_coef}

        if self.scanner_cache_capacity is None:
            self.scanner_cache_capacity = self.n_walkers

if __name__ == "__main__":
    CONFIG = PySCFInput()

    run(CONFIG)
