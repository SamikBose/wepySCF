"""Input configuration for CPU-only REVO/PySCF examples.

Edit this file instead of passing command-line arguments.
"""

import os
from dataclasses import dataclass


@dataclass
class PySCFInput:
    # System name and info
    system: str = "alanine"
    backend: str = "gpu"
    gpu_fallback_cpu_on_error: bool = False

    # Simulation size
    n_walkers: int = 5
    n_cycles: int = 5
    segment_length: int = 1

    # Walker initialization
    jitter: float = 0.01
    seed: int = 13

    # PySCF runner parameters
    basis: str = "sto-3g"
    method: str = "RHF"
    # Allowed methods include RHF/UHF, RKS/UKS, MP2/DFMP2, and CCSD.
    xc: str | None = None
    step_size: float = 1e-4
    # 'steepest_descent' performs deterministic energy minimization.
    # 'langevin' adds thermal noise to approximate finite-temperature sampling.
    dynamics_mode: str = "langevin"
    temperature_kelvin: float = 300.0
    use_scf_scanner: bool = True
    density_grid_shape: tuple[int, int, int] = (10, 10, 10)

    # CPU walker-level parallelization
    # If None, defaults to n_walkers (i.e., one worker per walker when possible).
    num_workers: int | None = None
    # Read the OMP_NUM_THREADS environment variable (used for logging; set the value using export before running)
    _omp_threads_env_var: str | None = os.environ.get("OMP_NUM_THREADS")

    # Output control
    write_h5: bool = False # TODO: Add write control to other systems (just alanine right now)
    write_dash: bool = False
    h5_path: str = ""
    dash_path: str = ""
    overwrite: bool = True

    def __post_init__(self) -> None:
        """Set output paths; need to do this after initialization since we need to wait for parameters to be set."""
        filename_base = f"{self.system}_{self.backend}_{self.n_walkers}W_{self.n_cycles}C_{self.dynamics_mode}_{self._omp_threads_env_var}T"
        if not self.h5_path:
            self.h5_path = f"{filename_base}.wepy.h5"
        if not self.dash_path:
            self.dash_path = f"{filename_base}.dash.org"


@dataclass
class WaterDimerInput(PySCFInput):
    # System name and info
    system: str = "waterdimer"

    # Simulation size
    n_walkers: int = 8
    n_cycles: int = 5
    segment_length: int = 2

    # Walker initialization
    jitter: float = 0.005

    # PySCF runner parameters
    method: str = "RHF"
    xc: str | None = "m06"


CONFIG = PySCFInput()
WATER_DIMER_RHF_CONFIG = WaterDimerInput(
    system="waterdimer_rhf",
    method="RHF",
)
WATER_DIMER_RKS_M06_CONFIG = WaterDimerInput(
    system="waterdimer_rks_m06",
    method="RKS",
    xc="m06",
)
