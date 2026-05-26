# Standard Library
import logging
logger = logging.getLogger(__name__)
import random as rand
import time
from copy import copy
from warnings import warn
import numpy as np
import mdtraj as mdj

from wepy.runners.runner import Runner
from wepy.walker import Walker, WalkerState

from wepy.work_mapper.task_mapper import WalkerTaskProcess # Haven't used these two but will them once
from wepy.work_mapper.worker import Worker                 # the PySCFRunner is fully established.
import cupy as cp


try:
    # Third Party Library
    import pyscf
    import pyscf.md
    from pyscf import gto
    #from pyscf.dft import rks

except ModuleNotFoundError:
    raise ModuleNotFoundError("PySCF has not been installed, which this runner requires...")

bohr_to_angstrom = 0.529177
angstrom_to_bohr = 1./bohr_to_angstrom


class PySCFRunner(Runner):

    def __init__(self,delta_T,segment_length,basis,platform=None,platform_kwargs=None,get_state_kwargs=None):
        """Constructor for PySCFRunner.

        Parameters
        ----------
        
        state : pyscf.md.NVE(scanner, dt=timestep, steps=n_steps).run() 
            The integrator run object which runs the simulation and eventually 
            stores the time-evolved simulation state.
            This is the container of the acceleration, velocity and the mole 
            object (molecular structure class in PySCF).
        
        delta_T : int or float.
            The timestep for the integrator in atomic unit.
        
        segment_length: int or float.
            The resampling time in number of steps.
            The numerical value that specifies how many dynamics steps are to be run.
        
        basis: str

        platform : str or None
            The specification for the default computational platform to use. 

        platform_kwargs : dict of str : bool, optional
            key-values to set for a platform with
            platform.setPropertyDefaultValue as the default for this
            runner.

        """

        self.delta_T = delta_T
        self.segment_length = segment_length
        self.basis = basis
        self.platform_name = platform
        self.platform_kwargs = platform_kwargs
        
        if get_state_kwargs is not None:
            for k in get_state_kwargs:
                self.getState_kwargs[k] = get_state_kwargs[k]


    def run_segment(self,walker,segment_length,platform='CUDA',platform_kwargs=None,**kwargs,):
        """Run dynamics for the walker.

        Parameters
        ----------
        walker : object implementing the PySCFWalker interface
            The walker for which dynamics will be propagated.
            This is ideally the walker provided to the runner after 
            the resampling and it would be run without pause for segment_length.

        segment_length : int or float
            The numerical value that specifies how many dynamics steps are to be run.

        platform : str or None 
            The specification for the default computational platform to use. 

        Returns
        -------
        new_walker : object implementing the PySCFWalker interface
            Walker after dynamics was run, only the state i.e., the 
            positions, velocities etc should be modified.
            The weight must not be modified.
        """
        import cupy as cp
        #from pyscf.dft import rks
        from pyscf import gto 
        import pyscf
        import pyscf.md
        #import os

        gpu_idx = int(platform_kwargs["DeviceIndex"]) # may not need the int
        print(f'Device idxs (GPU): {gpu_idx}')
        
        with cp.cuda.Device(gpu_idx):

            #os.environ["CUDA_VISIBLE_DEVICES"] = gpu_idx
            run_segment_start = time.time()

            free_mem = cp.cuda.runtime.memGetInfo()[0]
            print(f"Process {gpu_idx} is using GPU {cp.cuda.Device(gpu_idx).id}, which has {free_mem} B memory.")
            #import cupy
            #print(cupy.cuda.runtime.memGetInfo()[0])
            dt = self.delta_T
            
            #store the velocity from the walker state post resampling
            old_veloc = walker.state.velocities
            old_pos = walker.state.positions
            atom_sym = walker.state.elements
            
            # new velocity with small random noise
            random_noise = np.random.normal(loc=1.0, scale=0.00001, size=old_veloc.shape)
            updated_veloc = old_veloc*random_noise
            
            print(old_pos)
            print(old_veloc)
            print(updated_veloc)
            
            print('Check point A...') 
            
            mol = gto.Mole()
            mol.atom = list(zip(atom_sym, old_pos))
            mol.unit = 'Bohr'
            #mol.verbose = 4
            mol.max_memory = 900
            mol.basis = self.basis
            mol.build()
            print('Check point B...')
            
            mf = pyscf.scf.RHF(mol)
            mf.kernel()
            print('Check point C... HF built..,') 
            
            
            # scanner = grad.RHF(mf)  # This creates a gradient method
            grad_method = mf.nuc_grad_method()
            scanner = grad_method.as_scanner()
            scanner.mol = mol       # This is important for MD to work

            print('Check point D... Setting the scanner')

            integrator = pyscf.md.integrators.LangevinMiddle(scanner, T=100.0, friction_coef=1.0)  # timestep in fs
            # integrator = pyscf.md.integrators.Langevin(scanner, T=100.0, friction_coef=1.0)  # timestep in fs
            new_integrator = integrator.run(mol=scanner.mol, steps=3, dt=0.5)

            print('Check point E... MD has run!!')

            #mf = rks.RKS(mol, xc='LDA').density_fit()
            #scanner = mf.nuc_grad_method().as_scanner()
            #
            #scanner = hf.nuc_grad_method().as_scanner()
            #print('Check point D... Setting the scanner')

            #create the integrator
            print('Check 1')
            #integrator = pyscf.md.NVE(method=scanner, dt=self.delta_T, steps=segment_length, veloc=updated_veloc)
            ##create the pyscf simulation object using the integrator
            
            #print('Check 2')
            #new_integrator = integrator.kernel()
            ##integrator.run()
            #print('Check 3')
            new_mol = new_integrator.mol
            new_vel = new_integrator.veloc
            new_accel = new_integrator.accel

            # pass the data as a dictionary later TODO
            data = {'kin': new_integrator.ekin ,'pot': new_integrator.epot, 'temperature': new_integrator.temperature, 'time':new_integrator.time }
            
            new_pyscf_state = PySCFState(new_mol, new_vel, new_accel)
            new_walker = PySCFWalker(new_pyscf_state, walker.weight)
            
            run_segment_end = time.time()
            run_segment_time = run_segment_end - run_segment_start
            logger.info("Total internal run_segment time: {}".format(run_segment_time))
            print(walker, run_segment_time)
        
        return new_walker

    def pre_cycle(self, platform=None, platform_kwargs=None, **kwargs):
        # choose to use the platform spec in this function call or to
        # use the default one saved in the runner

        # if the platform is given locally use this one
        if platform is not None:
            logger.info(
                f"Setting the platform ({platform}) in the 'pre_cycle'"
                f"with platform kwargs: {platform_kwargs}"
            )
            # set the platform and kwargs for this cycle
            self._cycle_platform = platform
            self._cycle_platform_kwargs = platform_kwargs

        # otherwise we just don't set this and let resolution of
        # platform happen at run segment.

        super().pre_cycle(**kwargs)

        # each segment split times will get appended to this
        #self._last_cycle_segments_split_times = []
        pass

    def post_cycle(self, **kwargs):
        super().post_cycle(**kwargs)

        # remove the platform and kwargs for this cycle
        self._cycle_platform = None
        self._cycle_platform_kwargs = None

        pass


def gen_mol_state(positions, atom_symbols, basis):

    mol = gto.Mole()
    mol.atom = list(zip(atom_symbols, positions))
    mol.unit = 'Bohr'
    mol.basis = basis
    mol.max_memory = 5000
    mol.build()
    return mol

def gen_walker_state(mol, vel, accel):
    walker_state = PySCFState(mol, vel, accel)
    return walker_state

KEYS = ("positions","velocities","acceleration","elements")
class PySCFState(WalkerState):
    def __init__(self, mol_obj, vel, accel, **kwargs):
        self._data = {}
        for keys in KEYS:
            if keys == "positions":
                self._data[keys] = mol_obj.atom_coords()
            if keys == "velocities":
                self._data[keys] = vel
            if keys == "acceleration":
                self._data[keys] = accel
            if keys == "elements":
                self._data[keys] = mol_obj.elements

        self._sim_state = self._data
    def __getitem__(self, key):
        if key == "positions":
            return self._data["positions"]
        elif key == "velocities":
            return self._data["velocities"]
        elif key == "acceleration":
            return self._data["acceleration"]
        elif key == "elements":
            return self._data["elements"]

    @property
    def sim_state(self):
        return self._sim_state
    @property
    def positions(self):
        return self._data["positions"]
    @property
    def velocities(self):
        return self._data["velocities"]
    @property
    def acceleration(self):
        return self._data["acceleration"]
    @property
    def elements(self):
        return self._data["elements"]


class PySCFWalker(Walker):
    """Implementation of the Walker interface for PySCF package.
    A container for:
    - state (PySCF state, not the simulation state)
    - weight (WE weight)

    Important Note: This is where we introduce weights
    """
    def __init__(self, state, weight):
        """Constructor for Walker.
        Parameters
        ----------
        state : PySCFState object 
                The state should have state.position, state.velocities in place. 

        weight : float
                Weighted ensemble weight of each walker.
        """
        self.state = state
        self.weight = weight


class PySCFGPUWorker(Worker):
    """Worker for PySCF GPU simulations (CUDA platform supported only).
    
    This is intended to be used with the wepy.work_mapper.WorkerMapper
    work mapper class.

    This class must be used in order to ensure PySCF runs jobs on the
    appropriate GPU device.
    """

    NAME_TEMPLATE = "PySCFGPUWorker-{}"
    """The name template the worker processes are named to substituting in
    the process number."""

    def run_task(self, task):
        # get the platform
        platform = self.mapper_attributes["platform"]

        # get the device index from the attributes
        device_id = self.mapper_attributes["device_ids"][self._worker_idx]

        # make the platform kwargs dictionary
        platform_options = {"DeviceIndex": str(device_id)}

        logger.info(f"platform={platform}, platform_options={platform_options}")

        return task(
            platform=platform,
            platform_kwargs=platform_options,
        )

class PySCFGPUWalkerTaskProcess(WalkerTaskProcess):
    NAME_TEMPLATE = "PySCF_GPU_Walker_Task-{}"

    def run_task(self, task):
        logger.info(f"Starting to run a task as worker {self._worker_idx}")

        # get the platform
        platform = self.mapper_attributes["platform"]

        # get the device index from the attributes
        device_id = self.mapper_attributes["device_ids"][self._worker_idx]

        # make the platform kwargs dictionary
        platform_options = {"DeviceIndex": str(device_id)}

        logger.info(f"platform={platform}, platform_options={platform_options}")

        return task(
            platform=platform,
            platform_kwargs=platform_options,
        )
