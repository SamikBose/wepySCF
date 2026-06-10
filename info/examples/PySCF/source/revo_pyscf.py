"""Set up a REVO simulation with PySCF dynamics for alanine dipeptide.

This version uses a separate `pyscf_input.py` file for all PySCF/simulation parameters.
"""

# Set the default number of threads before importing libraries
import argparse
import os
import os.path as osp
import pickle
from copy import deepcopy
from glob import glob

from wepy.reporter.walker_pkl import WalkerPklReporter

os.environ.setdefault("OMP_NUM_THREADS", "1")  # Good default for PySCF CPU runs, but can be overridden by the user

# Standard Library
import importlib.util
import uuid
from time import perf_counter

# Third Party Library
import mdtraj as mdj
import numpy as np
import pyscf.gto as pyscf_gto
import pyscf.md as pyscf_md
from pyscf.data.nist import BOHR

# First Party Library
from pyscf_input import CONFIG

from wepy.boundary_conditions.boundary import NoBC
from wepy.reporter.dashboard import DashboardReporter
from wepy.reporter.pyscf import PySCFHDF5Reporter, PySCFRunnerDashboardSection
from wepy.resampling.resamplers.resampler import NoResampler
from wepy.resampling.resamplers.revo import REVOResampler
from wepy.runners.pyscf import PySCFCPUWorkerMapper, PySCFGPUWorkerMapper, PySCFRunner, PySCFState, PySCFWalker
from wepy.sim_manager import Manager
from wepy.util.mdtraj import mdtraj_to_json_topology


def parse_with_mdtraj_topology(topology_file_path: str):
    if topology_file_path.endswith(".pdb"):
        traj = mdj.load_pdb(topology_file_path)
    elif topology_file_path.endswith(".xyz"):
        traj = mdj.load_xyz(topology_file_path)
    else:
        raise RuntimeError("Must be a pdb or xyz file.")

    topology = traj.topology
    # mdtraj stores positions in nm, convert to Angstrom, then Bohr (atomic units)
    positions = np.asarray(traj.xyz[0], dtype=float) * 10.0 / BOHR
    symbols = [atom.element.symbol for atom in topology.atoms]

    return topology, symbols, positions


def build_mol(symbols, positions, basis, charge, spin):
    atom = [(symbol, tuple(coord)) for symbol, coord in zip(symbols, positions, strict=True)]

    return pyscf_gto.M(
        atom=atom,
        basis=basis,
        charge=charge,
        spin=spin,
        unit="Bohr",
    )


def generate_initial_walkers(symbols, positions, n_walkers, density_grid_shape):
    weight = 1.0 / n_walkers

    density_kwargs = {}
    if density_grid_shape is not None:
        density_kwargs = {
            "density_matrix": None,
            "density_grid": np.zeros(density_grid_shape),
            "density_grid_origin": np.zeros(3),
            "density_grid_spacing": np.ones(3),
        }

    mol = build_mol(symbols, positions, CONFIG.basis, CONFIG.charge, CONFIG.spin)

    velocities = (
        pyscf_md.distributions.MaxwellBoltzmannVelocity(mol, CONFIG.temperature_kelvin)
        if CONFIG.initialize_velocities
        else np.zeros_like(positions)
    )

    return [
        PySCFWalker(
            PySCFState(
                walker_id=str(uuid.uuid4()),
                symbols=symbols,
                mol=deepcopy(mol),
                positions=positions,
                velocities=velocities.copy(),
                accelerations=None,
                # Store as 1D feature arrays so the HDF5 reporter can extend them
                temperature=np.array([CONFIG.temperature_kelvin], dtype=float),
                total_energy=np.array([np.nan], dtype=float),
                potential=np.array([np.nan], dtype=float),
                kinetic=np.array([np.nan], dtype=float),
                **density_kwargs,
            ),
            weight,
        )
        for _ in range(n_walkers)
    ]


def build_revo_resampler(init_state):
    if CONFIG.resampler_parameters is None:
        return NoResampler()

    return REVOResampler(
        distance=CONFIG.distance,
        init_state=init_state,
        merge_dist=CONFIG.resampler_parameters.merge_dist,
        char_dist=CONFIG.resampler_parameters.char_dist,
        pmin=CONFIG.resampler_parameters.pmin,
        pmax=CONFIG.resampler_parameters.pmax,
    )


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--sub-step",
        type=int,
        default=None,
        help="Resume from a previous sub-step. Defaults to new simulation.",
        # TODO: Better description
    )
    parser.add_argument(
        "--from-branch",
        type=int,
        default=None,
        help="Branch of the previous sub-step to resume from. Defaults to latest.",
    )
    return parser.parse_args()


def main():
    # build_mapper
    if CONFIG.backend == "cpu":
        mapper = PySCFCPUWorkerMapper(num_workers=CONFIG.n_walkers)
    elif CONFIG.backend == "gpu":
        if importlib.util.find_spec("cupy") is None:
            raise SystemExit(
                "GPU backend requested but CuPy is not installed. "
                "Install a CUDA-matched CuPy package (e.g. cupy-cuda12x) "
                "or rerun with CPU.",
            )

        cuda_visible_devices = CONFIG._cuda_visible_devices_env_var  # noqa: SLF001
        if cuda_visible_devices != "":
            num_available = CONFIG._num_gpus_visible  # noqa: SLF001
            print(f"Found {num_available} available devices.")
        else:
            raise RuntimeError("No GPUs available: CUDA_VISIBLE_DEVICES is not set or empty.")

        device_ids = [i % num_available for i in range(CONFIG.n_walkers)]  # Round-robin assign workers to GPUs
        mapper = PySCFGPUWorkerMapper(num_workers=CONFIG.n_walkers, platform="CUDA", device_ids=device_ids)

    mdj_top, symbols, positions = parse_with_mdtraj_topology(CONFIG.topology_file_path)

    args = parse_args()

    # def get_sub_dir(base_dir, sub_step, branch=None, create=False):
    #     """Get the directory for a substep."""
    #     if branch is not None: # Reuse branch
    #         path = osp.join(base_dir, f"sub_{sub_step}_branch_{branch}")
    #     elif not create or not osp.isdir(osp.join(base_dir, f"sub_{sub_step}")):
    #         path = osp.join(base_dir, f"sub_{sub_step}")
    #     else:
    #         existing_dirs = glob(osp.join(base_dir, f"sub_{sub_step}_branch_*"))
    #         next_branch_num = (
    #             max(int(osp.basename(p).rsplit("_", 1)[-1]) for p in existing_dirs) + 1 if existing_dirs else 1
    #         )
    #         path = osp.join(base_dir, f"sub_{sub_step}_branch_{next_branch_num}")

    #     if create: # Create directory if needed
    #         os.makedirs(path)
    #     return path

    # target_cycle = args.sub_step * CONFIG.n_cycles  # Target cycle to start at
    # if args.sub_step == 0:
    #     print("Starting new simulation.")

    #     walkers = generate_initial_walkers(
    #         symbols=symbols,
    #         positions=positions,
    #         n_walkers=CONFIG.n_walkers,
    #         density_grid_shape=CONFIG.density_grid_shape,
    #     )

    #     # # Make the directory to store the log files if it doesn't exist
    #     # if CONFIG.write_h5 or CONFIG.write_dash or CONFIG.store_pickles:
    #     #     output_directory = CONFIG.output_directory

    #     #     # Append a number if the directory already exists
    #     #     if not CONFIG.overwrite and osp.isdir(output_directory):
    #     #         i = 1
    #     #         while osp.isdir(f"{output_directory}_{i}"):
    #     #             i += 1
    #     #         CONFIG.output_directory = output_directory = f"{output_directory}_{i}"
    #     #         print(f"Warning: output directory already exists, creating new directory: {output_directory}/")

    #     #     os.makedirs(output_directory, exist_ok=CONFIG.overwrite)
    # else:
    #     print(f"Restarting simulation at sub-step {args.sub_step}.")

    #     # base_directory = CONFIG.output_directory

    #     # # Find existing output directorys
    #     # candidates = [d for d in [base_directory, *glob(f"{base_directory}_*")] if osp.isdir(d)]
    #     # if not candidates:
    #     #     raise FileNotFoundError(f"Can't restart simulation: couldn't find output directory {base_directory}")

    #     # # Get the latest directory (highest number at the end)
    #     # latest_directory = max(
    #     #     candidates,
    #     #     key=lambda d: int(d.removeprefix(base_directory + "_")) if d != base_directory else 0,
    #     # )
    #     # CONFIG.output_directory = latest_directory
    #     # print(f"Using latest output directory: {latest_directory}/")

    #     # # Check if pkls directory exists
    #     # pkl_dir = osp.join(latest_directory, "pkls")
    #     # if not osp.isdir(pkl_dir):
    #     #     raise FileNotFoundError(f"Can't restart simulation: no pkls directory found in {latest_directory}")

    #     # # Load walkers from sub-step target cycle
    #     # target_cycle_idx = target_cycle - 1
    #     # target_pkl = osp.join(pkl_dir, f"walkers_cycle_{target_cycle_idx}.pkl")
    #     # if not osp.exists(target_pkl):
    #     #     raise FileNotFoundError(f"Can't restart simulation: expected pickle not found: {target_pkl}")
    #     # print(f"Resuming from pickle: {target_pkl}")

    #     # Restore the walkers from the pkl
    #     # with open(target_pkl, "rb") as f:
    #     #     walkers = pickle.load(f)  # noqa: S301

    def get_next_dir_num(base_dir: str) -> int:
        """Get the next directory number given a base directory name."""
        dirs = glob(f"{base_dir}_*")
        return max(int(osp.basename(d).rsplit("_", 1)[-1]) for d in dirs) + 1 if dirs else 1

    output_sub_directory = ""
    if args.sub_step is None:
        print("Starting new simulation.")
        start_cycle = None

        if CONFIG.overwrite:
            print(f"Warning: output directory already exists, overwriting: {CONFIG.output_directory}/")
        # Create new directory if not overwriting and it exists already
        elif osp.isdir(CONFIG.output_directory):
            new_dir_num = get_next_dir_num(CONFIG.output_directory)
            CONFIG.output_directory = f"{CONFIG.output_directory}_{new_dir_num}"
            print(f"Warning: output directory already exists, creating new directory: {CONFIG.output_directory}/")
            # Directory creation is handled downstream, we just need to set the right value

        walkers = generate_initial_walkers(
            symbols=symbols,
            positions=positions,
            n_walkers=CONFIG.n_walkers,
            density_grid_shape=CONFIG.density_grid_shape,
        )
    else:
        print(f"Restarting simulation at sub-step {args.sub_step}.")
        start_cycle = args.sub_step * CONFIG.n_cycles

    runner = PySCFRunner(
        basis=CONFIG.basis,
        method=CONFIG.method,
        xc=CONFIG.xc,
        dt=CONFIG.dt,
        integrator_cls=CONFIG.integrator_cls,
        integrator_kwargs=CONFIG.integrator_kwargs,
        integrator_temperature_kelvin=CONFIG.temperature_kelvin,
        backend=CONFIG.backend,
        density_grid_shape=CONFIG.density_grid_shape,
        use_scanner_caching=CONFIG.use_scanner_caching,
    )

    resampler = build_revo_resampler(walkers[0].state)

    reporters = []
    output_mode = "w" if CONFIG.overwrite else "x"

    # Add the pickle reporter (pickles walkers at the end of every cycle)
    if CONFIG.store_pickles:
        reporters.append(
            WalkerPklReporter(
                save_dir=osp.join(CONFIG.output_directory, output_sub_directory, "pkls"),
                freq=1,
                num_backups=2,
                start_cycle=start_cycle,
            )
        )

    # Add density fields if needed
    h5_save_fields = PySCFHDF5Reporter.DEFAULT_SAVE_FIELDS
    if CONFIG.density_grid_shape is not None:
        # Omit `density_matrix` by default because its array shape depends on
        # the AO basis size and can be expensive to store. Could store this later.
        h5_save_fields += (
            # "density_matrix",
            "density_grid",
            "density_grid_origin",
            "density_grid_spacing",
        )

    # Add the logging reporters
    if CONFIG.write_h5:
        reporters.append(
            PySCFHDF5Reporter(
                save_fields=h5_save_fields,
                file_paths=[CONFIG.h5_path(output_sub_directory)],
                modes=[output_mode],
                topology=mdtraj_to_json_topology(mdj_top),
                resampler=resampler,
                boundary_conditions=NoBC(),
            )
        )
    if CONFIG.write_dash:
        reporters.append(
            DashboardReporter(
                file_paths=[CONFIG.dash_path(output_sub_directory)],
                modes=[output_mode],
                runner_dash=PySCFRunnerDashboardSection(runner=runner),
            )
        )

    # Create the manager
    sim_manager = Manager(
        walkers,
        runner=runner,
        work_mapper=mapper,
        resampler=resampler,
        boundary_conditions=NoBC(),
        reporters=reporters,
    )

    # Run the simulation
    time = perf_counter()
    end_walkers, _ = sim_manager.run_simulation(
        n_cycles=CONFIG.n_cycles,
        segment_lengths=CONFIG.segment_length,
    )
    total_time = perf_counter() - time

    print(
        f"\nCompleted REVO/PySCF {CONFIG.backend.upper()} run in {total_time:.3f} sec "
        f"({total_time / CONFIG.n_cycles:.3f} sec / cycle)",
    )
    print(
        f"{len(end_walkers)} walkers, {CONFIG.n_cycles} cycles * {CONFIG.segment_length} steps "
        f"({CONFIG.n_cycles * CONFIG.segment_length} total MD steps)",
    )
    print(
        f"System: {CONFIG.system}, Basis: {CONFIG.basis}, Method: {CONFIG.method}"
        + (f"/{CONFIG.xc}, " if CONFIG.xc is not None else ", ")
        + f"Integrator: {CONFIG._integrator_name}",  # noqa: SLF001
    )
    if CONFIG.backend == "cpu":
        print(f"CPU workers: {CONFIG.n_walkers}")
    elif CONFIG.backend == "gpu":
        print(f"GPUs: {CONFIG._num_gpus_visible}")  # noqa: SLF001
        print(f"CUDA devices: [{CONFIG._cuda_visible_devices_env_var}]")  # noqa: SLF001
        # TODO: Does OpenMP threads affect when backend is GPU?
    print(f"OpenMP threads: {CONFIG._omp_threads_env_var}")  # noqa: SLF001
    temperatures = [walker.state.get("temperature").item() for walker in end_walkers]
    potentials = [walker.state.get("potential").item() for walker in end_walkers]
    kinetics = [walker.state.get("kinetic").item() for walker in end_walkers]
    energies = [p + k for p, k in zip(potentials, kinetics, strict=True)]
    print("Final walker temperatures:", temperatures)
    print("Final walker energies:", energies)
    print("Final walker potentials:", potentials)
    print("Final walker kinetics:", kinetics)
    print(f"Velocities initialized: {CONFIG.initialize_velocities}, Scanner caching: {CONFIG.use_scanner_caching}")


if __name__ == "__main__":
    main()
