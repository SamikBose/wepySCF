"""Input configuration for alanine dipeptide."""

# Standard Library
from pathlib import Path

# First Party Library
from pyscf.md.integrators import LangevinMiddle

# Third Party Library
from distance_metrics import qm_grid_density
from pyscf_input import PySCFInput
from revo_pyscf import run

from wepy.boundary_conditions.boundary import NoBC

CONFIG = PySCFInput(
    #
    # System
    #
    topology_file_path=str(Path(__file__).resolve().parent / "alanine_dipeptide.pdb"),
    system_name="alanine",
    #
    # Simulation parameters
    #
    backend="gpu",
    n_walkers=4,
    n_cycles=5,
    segment_length=10,
    #
    # PySCF runner parameters
    #
    basis="sto-3g",
    method="RHF",
    xc=None,
    charge=0,
    spin=0,
    dt=21,
    temperature_kelvin=300.0,
    density_grid_shape=(10, 10, 10),
    use_density_fitting=False,
    auxbasis="def2-universal-jkfit",
    #
    # PySCF integrator and any kwargs passed to it
    #
    integrator_cls=LangevinMiddle,
    integrator_kwargs={"friction_coef": 1.0},
    #
    # Distance metric and resampler parameters
    #
    distance_metric=qm_grid_density(),
    resampler_parameters=PySCFInput.ResamplerParameters(
        merge_dist=0.025,
        char_dist=0.1,
        pmin=1e-12,
        pmax=0.99,
    ),
    #
    # Boundary conditions
    #
    boundary_conditions=NoBC(),
    #
    # Misc
    #
    initialize_velocities=True,  # Initialize velocities from Maxwell-Boltzmann distribution (False uses zeros)
    use_scanner_caching=True,  # Cache scanners from the previous cycle to speed up first step greatly
    scanner_cache_capacity=None,  # The amount of scanners the cache can hold (None uses n_walkers)
    #
    # Output control
    #
    write_h5=True,
    write_dash=True,
    store_pickles=True,
    overwrite=False,
)

if __name__ == "__main__":
    run(CONFIG)
