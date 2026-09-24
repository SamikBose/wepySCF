"""Example simulation for SN2.

Run examples:
    python sn2_testing.py
    python sn2_testing.py --sub-step 1
    python sn2_testing.py --sub-step 1 --from-branch 2
"""

# Set the default number of threads before importing libraries to avoid oversubscription
from os import environ

environ.setdefault("OMP_NUM_THREADS", "1")
# Standard Library
from pathlib import Path

# Third Party Library
from pyscf.data.nist import BOHR
from pyscf.md.integrators import LangevinMiddle

# First Party Library
from wepy.resampling.distances.pyscf import ProtonTransferDistance
from wepy_tools.sim_makers.pyscf.cli import parse_args
from wepy_tools.sim_makers.pyscf.config import PySCFSimMakerConfig
from wepy_tools.sim_makers.pyscf.sim_maker import PySCFSimMaker

BREAK_PAIR = (0, 1)
MAKE_PAIR = (0, 5)
MAKE_CUTOFF_BOHR = 1.4 / BOHR
BREAK_CUTOFF_BOHR = 2.75 / BOHR


if __name__ == "__main__":
    args = parse_args()

    config = PySCFSimMakerConfig(
        # System
        topology_file_path=str(Path(__file__).resolve().parent / "sn2_opt.pdb"),
        system_name="SN2_testing",
        # Simulation parameters
        backend="GPU",
        n_walkers=4,
        n_cycles=5,
        segment_length=1,
        # PySCF runner parameters
        basis="sto-3g",
        method="RHF",
        charge=-1,
        dt=21,
        temperature_kelvin=100.0,
        # PySCF integrator and kwargs passed to it
        integrator_cls=LangevinMiddle,
        integrator_kwargs={"friction_coef": 1.0},
        # Distance metric and resampler parameters
        distance_metric=ProtonTransferDistance(break_pair=BREAK_PAIR, make_pair=MAKE_PAIR),
        resampler_parameters=PySCFSimMakerConfig.ResamplerParameters(
            merge_dist=0.05,
            char_dist=0.1,
            pmin=1e-12,
            pmax=0.20,
        ),
        # Boundary conditions
        use_boundary_conditions=True,
        break_pairs=[BREAK_PAIR],
        break_cutoffs=[BREAK_CUTOFF_BOHR],
        make_pairs=[MAKE_PAIR],
        make_cutoffs=[MAKE_CUTOFF_BOHR],
        # Initialization
        initialize_velocities=True,
        unique_initial_velocities=True,
        # Performance
        use_density_fitting=False,
        use_scanner_caching=True,
        # Output control
        write_h5=False,
        write_dash=False,
        store_pickles=False,
        overwrite=False,
    )

    sim_maker = PySCFSimMaker(config)
    sim_maker.run(sub_step=args.sub_step, from_branch=args.from_branch)
