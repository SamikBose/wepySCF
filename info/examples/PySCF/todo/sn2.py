"""Input configuration for alanine dipeptide."""

# Third Party Library
import pyscf.md as pyscf_md

# First Party Library
import .distance_metrics
from .pyscf_input import PySCFInput
from .revo_pyscf import run
from wepy.boundary_conditions.bond_distance import BondDistanceBC

CONFIG = PySCFInput(
    #
    # System
    #
    topology_file_path="./info/examples/PySCF/source/alanine_dipeptide.pdb",
    system_name="alanine",
    #
    # Simulation parameters
    #
    backend="gpu",
    n_walkers=4,
    n_cycles=5,
    segment_length=1, # TODO: Make 10
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
    integrator_cls=pyscf_md.integrators.LangevinMiddle,
    integrator_kwargs={"friction_coef": 1.0},
    #
    # Distance metric and resampler parameters
    #
    distance_metric=distance_metrics.qm_grid_density(),
    resampler_parameters=PySCFInput.ResamplerParameters(
        merge_dist=0.025,
        char_dist=0.1,
        pmin=1e-12,
        pmax=0.99,
    ),
    #
    # Boundary conditions
    #
    boundary_conditions=BondDistanceBC(
        # FIXME: auto gen initial states variable here?
        initial_states=[walker.state for walker in walkers],
        # topology=json_top,
        break_pairs=[(0, 1)],  # TODO: Use break/make pair from input file
        break_cutoffs=[0.5],  # nm
        make_pairs=[(0, 5)],
        make_cutoffs=[0.15],  # nm
    ),
    #
    # Misc
    #
    initialize_velocities=True,  # Initialize velocities from Maxwell-Boltzmann distribution (False uses zeros)
    use_scanner_caching=False,  # Cache scanners from the previous cycle to speed up first step greatly
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
