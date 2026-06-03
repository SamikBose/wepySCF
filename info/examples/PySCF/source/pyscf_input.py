"""Input configuration for CPU-only REVO/PySCF examples.

Edit this file instead of passing command-line arguments.
"""

import os
from dataclasses import dataclass, field

import pyscf.md as pyscf_md


@dataclass
class PySCFInput:
    # System name and info
    system: str = "alanine"
    backend: str = "gpu"

    # Simulation size
    n_walkers: int = 5
    n_cycles: int = 5
    segment_length: int = 1

    # PySCF runner parameters
    basis: str = "sto-3g"
    method: str = "RHF"
    # Allowed methods include RHF/UHF, RKS/UKS, MP2/DFMP2, and CCSD.
    xc: str | None = None
    dt: float = 21.0
    temperature_kelvin: float = 300.0
    density_grid_shape: tuple[int, int, int] = (10, 10, 10)

    # Select the PySCF MD integrator class and any kwargs passed to it.
    integrator_cls: type = pyscf_md.integrators.LangevinMiddle
    integrator_kwargs: dict = field(default_factory=lambda: {"friction_coef": 0.1})

    # CPU walker-level parallelization
    # If None, defaults to n_walkers (i.e., one worker per walker when possible).
    num_workers: int | None = None
    # Read the OMP_NUM_THREADS environment variable (used for logging; user sets the value before running)
    _omp_threads_env_var: str | None = os.environ.get("OMP_NUM_THREADS")

    # Output control
    write_h5: bool = False  # TODO: Add write control to other systems (just alanine right now)
    write_dash: bool = False
    h5_path: str | None = None
    dash_path: str | None = None
    overwrite: bool = True

    def __post_init__(self) -> None:
        """Set output paths; need to do this after initialization since we need to wait for parameters to be set."""
        integrator_name = getattr(self.integrator_cls, "__name__", "integrator")
        filename_base = (
            f"{self.system}_{self.backend}_{self.n_walkers}W_{self.n_cycles}C_"
            f"{integrator_name}_{self._omp_threads_env_var}T"
        )
        if not self.h5_path:
            self.h5_path = f"{filename_base}.wepy.h5"
        if not self.dash_path:
            self.dash_path = f"{filename_base}.dash.org"


# TODO: Update to match alanine
@dataclass
class WaterDimerInput(PySCFInput):
    # System name and info
    system: str = "waterdimer"

    # Simulation size
    n_walkers: int = 8
    n_cycles: int = 5
    segment_length: int = 2

    # Walker initialization
    jitter: float = 0.005  # TODO: Remove

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
