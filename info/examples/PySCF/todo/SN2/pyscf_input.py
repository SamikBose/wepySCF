"""Input configuration for REVO/PySCF examples."""

import os
from dataclasses import dataclass, field

import pyscf.md as pyscf_md
from wepy.resampling.distances.pyscf import DihedralDistance, ProtonTransferDistance, QMGridDensityDistance


@dataclass
class ResamplerParameters:
    merge_dist: float = 0.025
    char_dist: float = 0.01
    pmin: float = 1e-12
    pmax: float = 0.2


@dataclass
class PySCFInput:
    # System
    topology_file_path: str = "/mnt/research/PTR_bose/new_samik/run_sim/malonaldehyde/malonaldehyde_opt.pdb"
    system: str = "malonaldehyde"
    backend: str = "gpu"

    # Simulation size
    n_walkers: int = 24
    n_cycles: int = 100
    segment_length: int = 10

    # PySCF / QM  -- charge & spin now have annotations, so they are real fields
    basis: str = "def2-TZVP"
    method: str = "RKS"
    xc: str | None = "wb97x-d3bj"
    charge: int = 0
    spin: int = 0
    dt: int = 21
    temperature_kelvin: float = 200.0
    friction_coef: float = 1.0           # real field now (annotated)
    density_grid_shape: tuple[int, int, int] | None = None

    integrator_cls: type = pyscf_md.integrators.LangevinMiddle
    # Leave empty here; built from self.friction_coef in __post_init__ (as a float!)
    integrator_kwargs: dict = field(default_factory=dict)

    # Distance metric
    distance: object = field(
        default_factory=lambda: ProtonTransferDistance(break_pair=(7, 8), make_pair=(0, 8)))
    
   # Resampler
    resampler_parameters: ResamplerParameters | None = field(default_factory=ResamplerParameters)

    # Behaviour
    initialize_velocities: bool = True
    use_scanner_caching: bool = False
    if use_scanner_caching: scanner_tag = 'scanner_cache'
    else: scanner_tag = 'no_scanner_cache'

    # Output control
    write_h5: bool = True
    write_dash: bool = True
    store_pickles: bool = True
    overwrite: bool = False

    # ---- Derived values: declared with placeholders, filled in __post_init__ ----
    output_directory: str = ""
    filename_base: str = ""
    _integrator_name: str = ""
    _omp_threads_env_var: str = ""
    _cuda_visible_devices_env_var: str = ""
    _num_gpus_visible: int = 0

    def __post_init__(self) -> None:
        # integrator kwargs as a FLOAT, derived from the field
        if not self.integrator_kwargs:
            self.integrator_kwargs = {"friction_coef": self.friction_coef}

        self._integrator_name = getattr(self.integrator_cls, "__name__", "integrator")
        self._omp_threads_env_var = os.environ.get("OMP_NUM_THREADS", "")
        self._cuda_visible_devices_env_var = os.environ.get("CUDA_VISIBLE_DEVICES", "")
        self._num_gpus_visible = len(
            [x for x in self._cuda_visible_devices_env_var.split(",") if x.strip()]
        )

        # Reads self.* so it reflects ANY overrides, and pulls merge_dist from the
        # resampler itself -- no duplicate class-level merge_dist needed.
        md = self.resampler_parameters.merge_dist if self.resampler_parameters else "NA"
        self.output_directory = (
            f"{self.system}_{self.n_walkers}walkers_{self.n_cycles}cycles_"
            f"{self.segment_length}steps_{md}mergedist_{self.temperature_kelvin}Temp_"
            f"{self.dt}dt_{self._integrator_name}_{self.friction_coef}friction_{self.resampler_parameters.pmax}pmax_{self.scanner_tag}"
        )
        self.filename_base = f"{self.xc}_{self.basis}"

    def h5_path(self) -> str:
        return f"{self.output_directory}/{self.filename_base}.wepy.h5"

    def dash_path(self) -> str:
        return f"{self.output_directory}/{self.filename_base}.dash.org"


CONFIG = PySCFInput()
