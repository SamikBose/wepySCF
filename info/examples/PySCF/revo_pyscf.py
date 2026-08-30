"""Run a REVO simulation with PySCF dynamics.

This file should not be run on its own.
"""

# Standard Library
import argparse
import importlib.util
import os
import os.path as osp
import pickle
from copy import deepcopy
from glob import glob
from time import perf_counter

# Third Party Library
import mdtraj as mdj
import numpy as np
import pyscf.gto as pyscf_gto
import pyscf.md as pyscf_md
from pyscf.data.nist import BOHR

# First Party Library
from wepy.boundary_conditions.boundary import NoBC
from wepy.boundary_conditions.pyscf import PySCFBondDistanceBC
from wepy.reporter.dashboard import DashboardReporter
from wepy.reporter.pyscf import PySCFHDF5Reporter, PySCFRunnerDashboardSection
from wepy.reporter.walker_pkl import WalkerPklReporter
from wepy.resampling.resamplers.pyscf import PySCFREVOResampler
from wepy.resampling.resamplers.resampler import NoResampler
from wepy.runners.pyscf import PySCFCPUWorkerMapper, PySCFGPUWorkerMapper, PySCFRunner, PySCFState, PySCFWalker
from wepy.sim_manager import Manager
from wepy.util.mdtraj import mdtraj_to_json_topology


def parse_with_mdtraj_topology(topology_file_path: str):
    """Parse a pdb or xyz file using mdtraj and return the topology, symbols, and positions."""
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


def build_mol(symbols, positions, basis, charge, spin, ecp=None):
    atoms = [(symbol, tuple(coord)) for symbol, coord in zip(symbols, positions, strict=True)]
    return pyscf_gto.M(
        atom=atoms,
        basis=basis,
        ecp=ecp,
        charge=charge,
        spin=spin,
        unit="Bohr",
    )


def _generate_MB_velocities(mol, positions, temperature_kelvin, initialize_velocities):
    return (
        pyscf_md.distributions.MaxwellBoltzmannVelocity(mol, temperature_kelvin)
        if initialize_velocities
        else np.zeros_like(positions)
    )


def generate_initial_walkers(config, symbols, positions):
    weight = 1.0 / config.n_walkers

    density_kwargs = {}
    if config.density_grid_shape is not None:
        density_kwargs = {
            "density_grid": np.zeros(config.density_grid_shape),
        }

    mol = build_mol(symbols, positions, config.basis, config.charge, config.spin, config.ecp)
    if config.suppress_pyscf_output:
        mol.verbose = 0  # Suppress PySCF output

    shared_velocity = _generate_MB_velocities(mol, positions, config.temperature_kelvin, config.initialize_velocities)

    return [
        PySCFWalker(
            PySCFState(
                mol=deepcopy(mol),
                positions=positions,
                velocities=(
                    np.copy(shared_velocity)
                    if not config.unique_initial_velocities
                    else (
                        _generate_MB_velocities(mol, positions, config.temperature_kelvin, config.initialize_velocities)
                    )
                ),
                temperature=config.temperature_kelvin,
                **density_kwargs,
            ),
            weight,
        )
        for _ in range(config.n_walkers)
    ]


def build_revo_resampler(config, init_state):
    if config.resampler_parameters is None:
        return NoResampler()

    return PySCFREVOResampler(
        distance=config.distance_metric,
        init_state=init_state,
        merge_dist=config.resampler_parameters.merge_dist,
        char_dist=config.resampler_parameters.char_dist,
        pmin=config.resampler_parameters.pmin,
        pmax=config.resampler_parameters.pmax,
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


def run(config):
    # TODO: build_mapper
    if config.backend == "CPU":
        mapper = PySCFCPUWorkerMapper(num_workers=config.n_walkers)
    elif config.backend == "GPU":
        if importlib.util.find_spec("cupy") is None:
            raise SystemExit(
                "GPU backend requested but CuPy is not installed. "
                "Install a CUDA-matched CuPy package (e.g. cupy-cuda12x) "
                "or rerun with CPU.",
            )

        cuda_visible_devices = config._cuda_visible_devices_env_var  # noqa: SLF001
        if cuda_visible_devices != "":
            available_ids = [int(x) for x in cuda_visible_devices.split(",")]
            num_available = config._num_gpus_visible  # noqa: SLF001
            print(f"Found {num_available} available devices.")
        else:
            raise RuntimeError("No GPUs available: CUDA_VISIBLE_DEVICES is not set or empty.")

        # Round-robin assign workers to GPUs
        device_ids = [available_ids[i % num_available] for i in range(config.n_walkers)]

        mapper = PySCFGPUWorkerMapper(num_workers=config.n_walkers, platform="CUDA", device_ids=device_ids)

    mdj_top, symbols, positions = parse_with_mdtraj_topology(config.topology_file_path)

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

    # TODO: Move to function
    start_cycle = None
    output_directory = config.output_directory
    if args.sub_step is None:
        if args.from_branch is not None:
            raise ValueError("--from-branch is not supported without a sub-step specified.")

        print("Starting new simulation.")

        # Check base output directory first
        if config.write_h5 or config.write_dash or config.store_pickles:
            if osp.isdir(output_directory):
                if config.overwrite:
                    print(f"Warning: output directory already exists, overwriting: {output_directory}/")
                else:
                    # Create new directory if not overwriting and it exists already
                    next_dir_num = get_next_dir_num(output_directory)
                    output_directory = f"{output_directory}_{next_dir_num}"
                    print(f"Warning: output directory already exists, creating new directory: {output_directory}/")
            else:
                print(f"Creating output directory: {output_directory}/")

        walkers = generate_initial_walkers(
            config=config,
            symbols=symbols,
            positions=positions,
        )
    else:  # Sub-step provided
        # Resolve to the latest versioned base directory
        output_directory = get_latest_dir(output_directory)

        if args.sub_step == 0:
            if args.from_branch is not None:
                raise ValueError("--from-branch is not supported with sub-step 0.")
            print("Starting simulation in sub-step mode.")
        else:
            # Base directory must already exist in sub-step mode
            if not osp.isdir(output_directory):
                raise FileNotFoundError(f"Output directory does not exist: {output_directory}/")
            print(f"Continuing simulation at sub-step {args.sub_step}.")

        # Get next sub directory and previous sub directory
        sub_directory = f"sub_{args.sub_step}"
        prev_sub_directory = f"sub_{args.sub_step - 1}"
        if args.from_branch is not None:
            sub_directory += f"_branch_{args.from_branch}"
            prev_sub_directory += f"_branch_{args.from_branch}"
        prev_pkls_directory = osp.join(output_directory, prev_sub_directory, "pkls")

        # Output directory already exists
        if osp.isdir(osp.join(output_directory, sub_directory)):
            if config.overwrite:
                output_directory = osp.join(output_directory, sub_directory)
                print(f"Warning: sub-step output directory already exists, overwriting: {output_directory}/")
            else:
                if args.from_branch is not None:
                    raise ValueError(
                        f"Sub-step directory already exists and overwrite is disabled: "
                        f"{osp.join(output_directory, sub_directory)}"
                    )

                # Create new directory if not overwriting and it exists already
                next_branch_num = get_next_dir_num(osp.join(output_directory, f"sub_{args.sub_step}_branch"))
                output_directory = osp.join(output_directory, f"sub_{args.sub_step}_branch_{next_branch_num}")
                print(f"Warning: sub-step output directory already exists, creating new directory: {output_directory}/")
        else:
            output_directory = osp.join(output_directory, sub_directory)
            print(f"Creating sub-step output directory: {output_directory}/")

        # Load or generate walkers
        if args.sub_step == 0:
            walkers = generate_initial_walkers(
                config=config,
                symbols=symbols,
                positions=positions,
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

    if not osp.exists(output_directory):
        os.makedirs(output_directory)

    runner = PySCFRunner(
        backend=config.backend,
        method=config.method,
        xc=config.xc,
        population_method=config.population_method,
        dt=config.dt,
        integrator_cls=config.integrator_cls,
        integrator_kwargs=config.integrator_kwargs,
        integrator_temperature_kelvin=config.temperature_kelvin,
        density_grid_shape=config.density_grid_shape,
        use_density_fitting=config.use_density_fitting,
        auxbasis=config.auxbasis,
        use_scanner_caching=config.use_scanner_caching,
        scanner_cache_capacity=config.scanner_cache_capacity,
    )

    resampler = build_revo_resampler(config, walkers[0].state)

    boundary_conditions = (
        NoBC()
        if not config.use_boundary_conditions
        else PySCFBondDistanceBC(
            initial_states=[walker.state for walker in walkers],
            break_pairs=config.break_pairs,
            break_cutoffs=config.break_cutoffs,
            make_pairs=config.make_pairs,
            make_cutoffs=config.make_cutoffs,
        )
    )

    json_top = mdtraj_to_json_topology(mdj_top)

    reporters = []
    output_mode = "w" if config.overwrite else "x"

    # Add the pickle reporter (pickles walkers at the end of every cycle)
    if config.store_pickles:
        reporters.append(
            WalkerPklReporter(
                save_dir=osp.join(output_directory, "pkls"),
                freq=1,
                num_backups=2,
                start_cycle=start_cycle,
            )
        )

    # Add density fields if needed
    h5_save_fields = PySCFHDF5Reporter.DEFAULT_SAVE_FIELDS
    if config.density_grid_shape is not None:
        # Omit `density_matrix` by default because its array shape depends on
        # the AO basis size and can be expensive to store. Could store this later.
        h5_save_fields += (
            # "density_matrix",
            "density_grid",
            "density_grid_origin",
            "density_grid_spacing",
        )

    # Add the logging reporters
    if config.write_h5:
        reporters.append(
            PySCFHDF5Reporter(
                save_fields=h5_save_fields,
                file_path=config.get_h5_path(output_directory),
                mode=output_mode,
                topology=json_top,
                resampler=resampler,
                boundary_conditions=boundary_conditions,
            )
        )
    if config.write_dash:
        reporters.append(
            DashboardReporter(
                file_path=config.get_dash_path(output_directory),
                mode=output_mode,
                runner_dash=PySCFRunnerDashboardSection(runner),
            )
        )

    # Create the manager
    sim_manager = Manager(
        walkers,
        runner=runner,
        work_mapper=mapper,
        resampler=resampler,
        boundary_conditions=boundary_conditions,
        reporters=reporters,
    )

    # Run the simulation
    time = perf_counter()
    end_walkers, _ = sim_manager.run_simulation(
        n_cycles=config.n_cycles,
        segment_lengths=config.segment_length,
    )
    total_time = perf_counter() - time

    print(
        f"\nCompleted REVO/PySCF {config.backend.upper()} run in {total_time:.3f} sec "
        f"({total_time / config.n_cycles:.3f} sec / cycle)",
    )
    print(
        f"{len(end_walkers)} walkers, {config.n_cycles} cycles * {config.segment_length} steps "
        f"({config.n_cycles * config.segment_length} total MD steps)",
    )
    print(
        f"System: {config.system_name}, Basis: {config.basis}, Method: {config.method}"
        + (f"/{config.xc}, " if config.xc is not None else ", ")
        + f"Integrator: {config._integrator_name}",  # noqa: SLF001
    )
    if config.backend == "CPU":
        print(f"CPU workers: {config.n_walkers}, OpenMP threads: {config._omp_threads_env_var}")  # noqa: SLF001
    elif config.backend == "GPU":
        print(f"GPUs: {config._num_gpus_visible}, CUDA devices: [{config._cuda_visible_devices_env_var}]")  # noqa: SLF001
    temperatures = [walker.state.get("temperature").item() for walker in end_walkers]
    potentials = [walker.state.get("potential").item() for walker in end_walkers]
    kinetics = [walker.state.get("kinetic").item() for walker in end_walkers]
    energies = [p + k for p, k in zip(potentials, kinetics, strict=True)]
    print("Final walker temperatures:", temperatures)
    print("Final walker energies:", energies)
    print("Final walker potentials:", potentials)
    print("Final walker kinetics:", kinetics)
    print(
        f"Velocities initialized: {config.initialize_velocities}, "
        f"Unique velocities: {config.unique_initial_velocities}, "
        f"Density fitting: {config.use_density_fitting}, "
        f"Scanner caching: {config.use_scanner_caching}",
    )
