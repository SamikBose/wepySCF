# Set the default number of threads before importing libraries
import os

os.environ.setdefault("OMP_NUM_THREADS", "1")  # Good default for PySCF CPU runs, but can be overridden by the user

# Standard Library
import logging
logger = logging.getLogger(__name__)
import random as rand
import time
from copy import copy
from warnings import warn
import numpy as np
import mdtraj as mdj
import pickle as pkl

from wepy.runners.runner import Runner
from wepy.walker import Walker, WalkerState

from wepy.work_mapper.task_mapper import WalkerTaskProcess # Haven't used these two but will them once
from wepy.work_mapper.worker import Worker                 # the PySCFRunner is fully established.

#########################################################################
# These modules are loaded for testing will be deleted before pushing ##
from wepy.resampling.resamplers.resampler import NoResampler
from wepy.resampling.resamplers.revo import REVOResampler
from wepy.resampling.distances.pyscf import ProtonTransferDistance
from wepy.util.mdtraj import mdtraj_to_json_topology
from wepy.work_mapper.task_mapper import TaskMapper
from wepy.sim_manager import Manager
#########################################################################

from pyscf_runner import gen_walker_state, gen_mol_state, PySCFRunner, PySCFState, PySCFWalker, PySCFGPUWorker, PySCFGPUWalkerTaskProcess
from wepy.reporter.pyscf import PySCFHDF5Reporter

import argparse
import sys
import pyscf
from pyscf import gto

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='WE simulation with PySCF runner.')
    parser.add_argument('pdb', type=str,
                        help='Path to the input PDB file.')
    parser.add_argument('--break-pair', type=int, nargs=2, required=True,
                        metavar=('I', 'J'),
                        help='Atom indices for the bond being broken (e.g. --break-pair 0 1).')
    parser.add_argument('--make-pair', type=int, nargs=2, required=True,
                        metavar=('K', 'L'),
                        help='Atom indices for the bond being formed (e.g. --make-pair 1 3).')
    parser.add_argument('--h5-path', type=str, default='wepy_pyscf.h5',
                        help='Output HDF5 file path (default: wepy_pyscf.h5).')
    parser.add_argument('--n-cycles', type=int, default=10,
                        help='Number of WE cycles to run (default: 10).')
    parser.add_argument('--n-walkers', type=int, default=1,
                        help='Number of walkers (default: 1).')
    parser.add_argument('--merge-dist', type=float, default=0.25,
                        help='REVO merge distance threshold (default: 0.25).')
    parser.add_argument('--char-dist', type=float, default=0.1,
                        help='REVO characteristic distance (default: 0.1).')
    args = parser.parse_args()

    bohr_to_angstrom = 0.529177
    angstrom_to_bohr = 1./bohr_to_angstrom

    pdb = mdj.load_pdb(args.pdb)

    atoms, _ = pdb.topology.to_dataframe()
    atom_symbols = []
    for element in atoms.element:
        atom_symbols.append(element)

    init_pos_bohr = pdb.xyz[0]*10*angstrom_to_bohr

    basis = 'sto-3g'
    mol = gen_mol_state(init_pos_bohr, atom_symbols,basis)

    hf = mol.RHF().run()

    #mf = rks.RKS(mol, xc='LDA').density_fit()
    #e_dft = mf.kernel()  # compute total energy
    #scanner = mf.nuc_grad_method().as_scanner()

    scanner = hf.nuc_grad_method().as_scanner()
    integrator = pyscf.md.integrators.LangevinMiddle(scanner, T=100.0, friction_coef=1.0, dt=21, steps=2)
    # integrator = pyscf.md.integrators.Langevin(scanner, T=100.0, friction_coef=1.0, dt=21, steps=2)
    ref_integrator = integrator.run()

    init_vel = ref_integrator.veloc
    init_accel = ref_integrator.accel

    init_walker_state = gen_walker_state(ref_integrator.mol, init_vel, init_accel)



    #init_state_filename = 'init_walker_state.pkl'
    #init_walker_state = pkl.load(open(init_state_filename, 'rb'))
    # separate script to get the walkerstate and save that in a pickle 
    # basically load the pkl in this code
    # TRY NOT TO init any GPU based object with CUDA!

    print('Initial PySCFState walker state generated...')

    n_walker = args.n_walkers
    init_wt = 1/n_walker
    walker_list = []
    for walker in range(n_walker):
        walker_list.append(PySCFWalker(state=init_walker_state, weight=init_wt))


    resampling_steps = 1
    basis = 'sto-3g'
    timestep=21
    # Building the PySCF runner object
    runner = PySCFRunner(delta_T=timestep, segment_length=resampling_steps, basis=basis, platform='CUDA')
    print('Done building the PySCFRunner...')

    distance = ProtonTransferDistance(break_pair=tuple(args.break_pair),
                                      make_pair=tuple(args.make_pair))

    # REVO
    resampler = REVOResampler(distance=distance,
                              init_state=init_walker_state,
                              weights=True,
                              pmax=0.1,
                              dist_exponent=4,
                              merge_dist=args.merge_dist,
                              char_dist=args.char_dist)
    #resampler = NoResampler()
    # never going to satisfy! :P
    #ubc = UnbindingBC(cutoff_distance=1.0,  # nm
    #                      initial_state=init_walker_state,
    #                      topology=json_top,
    #                      ligand_idxs=lig_idxs,
    #                      receptor_idxs=protein_idxs)

    json_topology = mdtraj_to_json_topology(pdb.topology)
    h5_reporter = PySCFHDF5Reporter(
        file_paths=[args.h5_path],
        modes=['w'],
        topology=json_topology,
        resampler=resampler,
    )

    # may break...

    # Guard: cupy/gpu4pyscf must NOT be imported before this point when using
    # proc_start_method='fork'. Forking after a CUDA context is created corrupts
    # GPU state in child processes. GPU initialization happens inside run_segment
    # (in the worker processes), never in the parent.
    if 'cupy' in sys.modules or 'gpu4pyscf' in sys.modules:
        raise RuntimeError(
            "cupy or gpu4pyscf was imported before the TaskMapper with proc_start_method='fork'. "
            "This will corrupt CUDA contexts in worker processes. "
            "Remove any top-level gpu4pyscf/cupy imports, or switch to proc_start_method='spawn'."
        )

    mapper = TaskMapper(walker_task_type=PySCFGPUWalkerTaskProcess,
                        proc_start_method='fork', num_workers=1,
                        platform='CUDA',
                        device_ids=[0])
                        # device_ids=[0, 1]) # Use multiple GPUs (also increase num workers accordingly)

    # build the simulation manager
    sim_manager = Manager(walker_list,
                          runner=runner,
                          resampler=resampler,
                          work_mapper=mapper,
                          reporters=[h5_reporter])

    #------------------------------
    # Run the simulation
    #------------------------------
    print('Sim manager is built...')
    n_steps = 1
    n_cycles = args.n_cycles

    # run a simulation with the manager for n_steps cycles of length 1000 each
    steps_list = [n_steps for i in range(n_cycles)]

    # and..... go!
    sim_manager.run_simulation(n_cycles,
                               steps_list)
