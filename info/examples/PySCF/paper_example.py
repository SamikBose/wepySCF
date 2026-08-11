##########################################################################
from pyscf.md.integrators import LangevinMiddle

from wepy.runners.pyscf import PySCFRunner

n_walkers = 24
temperature_kelvin = 100.0

runner = PySCFRunner(
    backend="GPU",
    method="RKS",
    xc="wb97x_v",
    population_method="meta-lowdin",
    dt=21,
    integrator_cls=LangevinMiddle,
    integrator_kwargs={"friction_coef": 1.0},
    integrator_temperature_kelvin=temperature_kelvin,
    density_grid_shape=None,
    use_density_fitting=True,
    auxbasis="aug-cc-pVDZ-jkfit",
    use_scanner_caching=True,
    scanner_cache_capacity=n_walkers,
)

##########################################################################
from copy import deepcopy

import mdtraj as mdj
import numpy as np
import pyscf.gto as pyscf_gto
from pyscf.data.nist import BOHR
from pyscf.md.distributions import MaxwellBoltzmannVelocity

from wepy.runners.pyscf import PySCFState, PySCFWalker

traj = mdj.load_pdb("sn2.pdb")

mdj_top = traj.topology
# mdtraj stores positions in nm, convert to Angstrom, then Bohr (atomic units)
positions = np.asarray(traj.xyz[0], dtype=float) * 10.0 / BOHR
symbols = [atom.element.symbol for atom in mdj_top.atoms]

weight = 1.0 / n_walkers

atoms = [(symbol, tuple(coord)) for symbol, coord in zip(symbols, positions, strict=True)]
mol = pyscf_gto.M(
    atom=atoms,
    basis="aug-cc-pVDZ",
    ecp=None,
    charge=-1,
    spin=0,
    unit="Bohr",
)

walkers = [
    PySCFWalker(
        PySCFState(
            mol=deepcopy(mol),
            positions=positions,
            velocities=MaxwellBoltzmannVelocity(mol, temperature_kelvin),
            temperature=temperature_kelvin,
        ),
        weight,
    )
    for _ in range(n_walkers)
]

##########################################################################
from wepy.resampling.distances.pyscf import ProtonTransferDistance
from wepy.resampling.resamplers.pyscf import PySCFREVOResampler

break_pair = (0, 1)
make_pair = (0, 5)
break_cutoff = 7.56  # Bohr (C-Br counted broken)
make_cutoff = 2.83  # Bohr (C-F counted formed)

resampler = PySCFREVOResampler(
    distance=ProtonTransferDistance(break_pair, make_pair),
    init_state=walkers[0].state,  # FIXME: Need to pass all the states since init velocities are different?
    merge_dist=0.05,
    char_dist=0.1,
    pmin=1e-12,
    pmax=0.20,
)

##########################################################################
from wepy.boundary_conditions.pyscf import PySCFBondDistanceBC

boundary_conditions = PySCFBondDistanceBC(
    initial_states=[walker.state for walker in walkers],
    # topology=json_top,
    break_pairs=[break_pair],
    break_cutoffs=[break_cutoff],
    make_pairs=[make_pair],
    make_cutoffs=[make_cutoff],
)

##########################################################################
from wepy.runners.pyscf import PySCFCPUWorkerMapper, PySCFGPUWorkerMapper

backend = "GPU"
device_ids = [0, 1, 2, 3]  # 4 GPUs

if backend == "CPU":
    mapper = PySCFCPUWorkerMapper(num_workers=n_walkers)
elif backend == "GPU":
    mapper = PySCFGPUWorkerMapper(num_workers=n_walkers, platform="CUDA", device_ids=device_ids)

##########################################################################
from wepy.sim_manager import Manager

sim_manager = Manager(
    walkers,
    runner=runner,
    work_mapper=mapper,
    resampler=resampler,
    boundary_conditions=boundary_conditions,
)

end_walkers, sim_components = sim_manager.run_simulation(
    n_cycles=10,
    segment_lengths=100,
)
