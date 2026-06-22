"""Input configuration for CPU-only REVO/PySCF examples."""

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
    use_density_fitting: bool = False
    auxbasis: str | None = "def2-universal-jkfit"

    #
    # Select the PySCF MD integrator class and any kwargs passed to it
    #
    integrator_cls: type = pyscf_md.integrators.LangevinMiddle
    integrator_kwargs: dict = field(default_factory=lambda: {"friction_coef": 1.0})

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
        merge_dist: float = 0.025
        char_dist: float = 0.1
        pmin: float = 1e-12
        pmax: float = 0.99

    # If resampler parameters is None, then no resampler is used
    resampler_parameters: ResamplerParameters | None = field(default_factory=ResamplerParameters)

    #
    # Misc
    #
    initialize_velocities: bool = True  # Initialize velocities from Maxwell Boltzmann distribution (False uses zeros)
    use_scanner_caching: bool = False  # Cache scanners from the previous cycle to speed up first step greatly

    #
    # Read only stuff for naming/logging
    #
    _integrator_name: str = getattr(integrator_cls, "__name__", "integrator")
    _omp_threads_env_var: str = os.environ.get("OMP_NUM_THREADS", "")
    _cuda_visible_devices_env_var: str = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    _num_gpus_visible = len([x for x in _cuda_visible_devices_env_var.split(",") if x.strip()])

    #
    # Output control
    #
    write_h5: bool = True
    write_dash: bool = True
    store_pickles: bool = True
    overwrite: bool = False

    output_directory = f"{system}_{n_walkers}W_{n_cycles}C_{segment_length}S_{_integrator_name}"
    filename_base = f"{backend}_{_omp_threads_env_var}T_{_num_gpus_visible}G"

    def h5_path(self) -> str:
        """Return the h5 path (evaluated at runtime)."""
        return f"{self.output_directory}/{self.filename_base}.wepy.h5"

    def dash_path(self) -> str:
        """Return the dash path (evaluated at runtime)."""
        return f"{self.output_directory}/{self.filename_base}.dash.org"


CONFIG = PySCFInput()
