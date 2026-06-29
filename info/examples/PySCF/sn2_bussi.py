"""Input configuration for alanine dipeptide."""

# Standard Library
from pathlib import Path

# First Party Library
from pyscf.md.integrators import NVTBussi

# Third Party Library
from distance_metrics import proton_transfer
from pyscf_input import PySCFInput
from revo_pyscf import run

BREAK_PAIR = (0, 1)
MAKE_PAIR = (0, 5)
BREAK_CUTOFF = 0.5  # nm
MAKE_CUTOFF = 0.15  # nm

CONFIG = PySCFInput(
    #
    # System
    #
    topology_file_path=str(Path(__file__).resolve().parent / "sn2.pdb"),
    system_name="sn2",
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
    basis="aug-cc-pVDZ",
    method="RKS",
    xc="wb97x_v",
    charge=-1,
    spin=0,
    dt=21,
    temperature_kelvin=300.0,
    density_grid_shape=None,
    use_density_fitting=True,
    auxbasis="def2-universal-jkfit",
    #
    # PySCF integrator and any kwargs passed to it
    #
    integrator_cls=NVTBussi,
    integrator_kwargs={"taut": 4134.0}, # a.u. of time (~0.1 ps)
    #
    # Distance metric and resampler parameters
    #
    distance_metric=proton_transfer(BREAK_PAIR, MAKE_PAIR),
    resampler_parameters=PySCFInput.ResamplerParameters(
        merge_dist=0.025,
        char_dist=0.1,
        pmin=1e-12,
        pmax=0.99,
    ),
    #
    # Boundary conditions
    #
    use_boundary_conditions=True,
    break_pairs=[BREAK_PAIR],
    break_cutoffs=[BREAK_CUTOFF],
    make_pairs=[MAKE_PAIR],
    make_cutoffs=[MAKE_CUTOFF],
    #
    # Misc
    #
    initialize_velocities=True,  # Initialize velocities from Maxwell-Boltzmann distribution (False uses zeros)
    unique_initial_velocities=True,  # Generate unique initial velocities for each walker
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
