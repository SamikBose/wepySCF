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
    #if CONFIG.resampler_parameters is None:
    #    return NoResampler()

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

    def get_next_dir_num(base_dir: str) -> int:
        """Get the next directory number given a base directory name."""
        dirs = [d for d in glob(f"{base_dir}_*") if osp.basename(d).rsplit("_", 1)[-1].isdigit()]
        return max(int(osp.basename(d).rsplit("_", 1)[-1]) for d in dirs) + 1 if dirs else 1

    def get_latest_dir(base_dir: str) -> str:
        """Get the latest versioned directory, or the base directory if no versioned ones exist."""
        dirs = [d for d in glob(f"{base_dir}_*") if osp.basename(d).rsplit("_", 1)[-1].isdigit()]
        if dirs:
            return max(dirs, key=lambda d: int(osp.basename(d).rsplit("_", 1)[-1]))
        return base_dir

    def get_last_pkl(directory: str) -> str:
        """Get the path to the last walker pkl file in a pkls directory."""
        pkls = glob(osp.join(directory, "walkers_cycle_*.pkl"))
        if not pkls:
            raise FileNotFoundError(f"No walker pkl files found in {directory}/")
        return max(pkls, key=lambda p: int(osp.basename(p).removeprefix("walkers_cycle_").removesuffix(".pkl")))

    # Directory creation is handled downstream, we just need to set the right value
    start_cycle = None
    if args.sub_step is None:
        if args.from_branch is not None:
            raise ValueError("--from-branch is not supported without a sub-step specified.")

        print("Starting new simulation.")

        # Check base output directory first
        if osp.isdir(CONFIG.output_directory):
            if CONFIG.overwrite:
                print(f"Warning: output directory already exists, overwriting: {CONFIG.output_directory}/")
            else:
                # Create new directory if not overwriting and it exists already
                next_dir_num = get_next_dir_num(CONFIG.output_directory)
                CONFIG.output_directory = f"{CONFIG.output_directory}_{next_dir_num}"
                print(f"Warning: output directory already exists, creating new directory: {CONFIG.output_directory}/")
        else:
            print(f"Creating output directory: {CONFIG.output_directory}/")

        walkers = generate_initial_walkers(
            symbols=symbols,
            positions=positions,
            n_walkers=CONFIG.n_walkers,
            density_grid_shape=CONFIG.density_grid_shape,
        )
    else:  # Sub-step provided
        # Resolve to the latest versioned base directory
        CONFIG.output_directory = get_latest_dir(CONFIG.output_directory)
        # Base directory must already exist in sub-step mode
        if not osp.isdir(CONFIG.output_directory):
            raise FileNotFoundError(f"Output directory does not exist: {CONFIG.output_directory}/")

        if args.sub_step == 0:
            if args.from_branch is not None:
                raise ValueError("--from-branch is not supported with sub-step 0.")
            print(f"Starting simulation in sub-step mode.")
        else:
            print(f"Continuing simulation at sub-step {args.sub_step}.")

        # Get next sub directory and previous sub directory
        sub_directory = f"sub_{args.sub_step}"
        prev_sub_directory = f"sub_{args.sub_step - 1}"
        if args.from_branch is not None:
            sub_directory += f"_branch_{args.from_branch}"
            prev_sub_directory += f"_branch_{args.from_branch}"
        prev_pkls_directory = osp.join(CONFIG.output_directory, prev_sub_directory, "pkls")

        # Output directory already exists
        if osp.isdir(osp.join(CONFIG.output_directory, sub_directory)):
            if CONFIG.overwrite:
                CONFIG.output_directory = osp.join(CONFIG.output_directory, sub_directory)
                print(f"Warning: sub-step output directory already exists, overwriting: {CONFIG.output_directory}/")
            else:
                if args.from_branch is not None:
                    raise ValueError(
                        f"Sub-step directory already exists and overwrite is disabled: "
                        f"{osp.join(CONFIG.output_directory, sub_directory)}"
                    )

                # Create new directory if not overwriting and it exists already
                next_branch_num = get_next_dir_num(osp.join(CONFIG.output_directory, f"sub_{args.sub_step}_branch"))
                CONFIG.output_directory = osp.join(
                    CONFIG.output_directory, f"sub_{args.sub_step}_branch_{next_branch_num}"
                )
                print(
                    f"Warning: sub-step output directory already exists, creating new directory: {CONFIG.output_directory}/"
                )
        else:
            CONFIG.output_directory = osp.join(CONFIG.output_directory, sub_directory)
            print(f"Creating sub-step output directory: {CONFIG.output_directory}/")

        # Load or generate walkers
        if args.sub_step == 0:
            walkers = generate_initial_walkers(
                symbols=symbols,
                positions=positions,
                n_walkers=CONFIG.n_walkers,
                density_grid_shape=CONFIG.density_grid_shape,
            )
        else:
            # Load walkers from the source directory
            target_pkl = get_last_pkl(prev_pkls_directory)
            print(f"Resuming from: {target_pkl}")

            # Calculate the start cycle from the pickle index
            start_cycle = (
                int(target_pkl.rsplit("_", 1)[-1].removesuffix(".pkl")) + 2
            )  # 0-based index so add 2 for next cycle

            # Restore the walkers from the pkl
            with open(target_pkl, "rb") as f:
                walkers = pickle.load(f)  # noqa: S301

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
                save_dir=osp.join(CONFIG.output_directory, "pkls"),
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
                file_paths=[CONFIG.h5_path()],
                modes=[output_mode],
                topology=mdtraj_to_json_topology(mdj_top),
                resampler=resampler,
                boundary_conditions=NoBC(),
            )
        )
    if CONFIG.write_dash:
        reporters.append(
            DashboardReporter(
                file_paths=[CONFIG.dash_path()],
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
