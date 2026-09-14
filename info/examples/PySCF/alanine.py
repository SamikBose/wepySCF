"""Example simulation for alanine dipeptide.

Run examples:
    python alanine.py
    python alanine.py --sub-step 1
    python alanine.py --sub-step 1 --from-branch 2
"""

# Set the default number of threads before importing libraries to avoid oversubscription
from os import environ

environ.setdefault("OMP_NUM_THREADS", "1")
# Standard Library
from pathlib import Path

# Third Party Library
from pyscf.md.integrators import LangevinMiddle

# First Party Library
from wepy.resampling.distances.pyscf import QMGridDensityDistance
from wepy_tools.sim_makers.pyscf.cli import parse_args
from wepy_tools.sim_makers.pyscf.config import PySCFSimMakerConfig
from wepy_tools.sim_makers.pyscf.sim_maker import PySCFSimMaker

if __name__ == "__main__":
    args = parse_args()

    config = PySCFSimMakerConfig(
        # System
        topology_file_path=str(Path(__file__).resolve().parent / "alanine_dipeptide.pdb"),
        system_name="Alanine",
        # Simulation parameters
        backend="GPU",
        n_walkers=4,
        n_cycles=5,
        segment_length=10,
        # PySCF runner parameters
        basis="sto-3g",
        method="RHF",
        dt=21,
        temperature_kelvin=300.0,
        density_grid_shape=(10, 10, 10),
        # PySCF integrator and kwargs passed to it
        integrator_cls=LangevinMiddle,
        integrator_kwargs={"friction_coef": 1.0},
        # Distance metric and resampler parameters
        distance_metric=QMGridDensityDistance(grid_key="density_grid", normalize=True),
        resampler_parameters=PySCFSimMakerConfig.ResamplerParameters(
            merge_dist=0.025,
            char_dist=0.1,
            pmin=1e-12,
            pmax=0.99,
        ),
        # Initialization
        initialize_velocities=True,
        unique_initial_velocities=True,
        # Performance
        use_density_fitting=False,
        use_scanner_caching=True,
        # Output control
        write_h5=True,
        write_dash=True,
        store_pickles=True,
        overwrite=False,
    )

    sim_maker = PySCFSimMaker(config)
    sim_maker.run(sub_step=args.sub_step, from_branch=args.from_branch)
