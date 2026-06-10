"""Input configuration for CPU-only REVO/PySCF examples.

Edit this file instead of passing command-line arguments.
"""

import os
from dataclasses import dataclass, field

import pyscf.md as pyscf_md

from wepy.resampling.distances.pyscf import ProtonTransferDistance, QMGridDensityDistance


@dataclass
class PySCFInput:
    #
    # System name and info
    #
    topology_file_path: str = "./info/examples/PySCF/source/alanine_dipeptide.pdb"
    system: str = "alanine"
    backend: str = "gpu"

    #
    # Simulation size
    #
    n_walkers: int = 4
    n_cycles: int = 5
    segment_length: int = 10

    #
    # PySCF runner parameters
    #
    basis: str = "sto-3g"
    method: str = "RHF"
    # Allowed methods include RHF/UHF, RKS/UKS
    xc: str | None = None
    charge: int = 0
    spin: int = 0
    dt: int = 21
    temperature_kelvin: float = 300.0
    # density_grid_shape: tuple[int, int, int] | None = None
    density_grid_shape: tuple[int, int, int] | None = (10, 10, 10)
    initialize_velocities: bool = True  # Initialize velocities from Maxwell Boltzmann distribution (False uses zeros)

    #
    # Select the PySCF MD integrator class and any kwargs passed to it
    #
    integrator_cls: type = pyscf_md.integrators.LangevinMiddle
    integrator_kwargs: dict = field(default_factory=lambda: {"friction_coef": 1e-5})

    #
    # Distance metric and resampler parameters
    #
    @staticmethod
    def distance_qm_grid_density():
        return QMGridDensityDistance(grid_key="density_grid", normalize=True)

    @staticmethod
    def distance_proton_transfer(break_pair: tuple[int, int], make_pair: tuple[int, int]):
        return ProtonTransferDistance(break_pair=break_pair, make_pair=make_pair)

    distance = distance_qm_grid_density()

    @dataclass
    class ResamplerParameters:
        merge_dist: float = 0.5
        char_dist: float = 0.1
        pmin: float = 1e-12
        pmax: float = 0.99

    # If resampler parameters is None, then no resampler is used
    resampler_parameters: ResamplerParameters | None = field(default_factory=ResamplerParameters)

    #
    # Output control
    #
    write_h5: bool = True
    write_dash: bool = True
    h5_path: str | None = None
    dash_path: str | None = None
    overwrite: bool = False

    #
    # Misc
    #
    initialize_velocities: bool = True  # Initialize velocities from Maxwell Boltzmann distribution (False uses zeros)
    use_scanner_caching: bool = False  # Cache scanners from the previous cycle to speed up first step greatly

    # Read the OMP_NUM_THREADS environment variable (used for logging; user sets the value before running)
    _omp_threads_env_var: str | None = field(default_factory=lambda: os.environ.get("OMP_NUM_THREADS", "unset"))

    def __post_init__(self) -> None:
        """Set output paths; need to do this after initialization since we need to wait for parameters to be set."""
        integrator_name = getattr(self.integrator_cls, "__name__", "integrator")
        # TODO: Don't use threads anymore in here? num gpus might be more helpful
        filename_base = (
            f"{self.system}_{self.backend}_{self.n_walkers}W_{self.n_cycles}C_"
            f"{integrator_name}_{self._omp_threads_env_var}T"
        )
        if not self.h5_path:
            self.h5_path = f"{filename_base}.wepy.h5"
        if not self.dash_path:
            self.dash_path = f"{filename_base}.dash.org"


CONFIG = PySCFInput()
