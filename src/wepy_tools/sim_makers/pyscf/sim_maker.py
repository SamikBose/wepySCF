"""PySCF wepy simulation maker and runner."""

# Standard Library
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
from pyscf.data.nist import AMU2AU, BOHR, BOLTZMANN, HARTREE2J

# First Party Library
from wepy.boundary_conditions.boundary import NoBC
from wepy.boundary_conditions.pyscf import PySCFBondDistanceBC, PySCFMultiBoundaryBC
from wepy.reporter.dashboard import DashboardReporter
from wepy.reporter.pyscf import PySCFHDF5Reporter, PySCFRunnerDashboardSection
from wepy.reporter.walker_pkl import WalkerPklReporter
from wepy.resampling.resamplers.pyscf import PySCFREVOResampler
from wepy.resampling.resamplers.resampler import NoResampler
from wepy.runners.pyscf import PySCFCPUWorkerMapper, PySCFGPUWorkerMapper, PySCFRunner, PySCFState, PySCFWalker
from wepy.sim_manager import Manager
from wepy.util.mdtraj import mdtraj_to_json_topology
from wepy_tools.sim_makers.pyscf.config import PySCFSimMakerConfig

AMU_TO_ELECTRON_MASS = AMU2AU
BOLTZMANN_HARTREE_PER_K = BOLTZMANN / HARTREE2J


class PySCFSimMaker:
    """Builds and runs a REVO/PySCF simulation from a PySCFSimMakerConfig."""

    def __init__(self, config: PySCFSimMakerConfig):
        self.config = config

    #
    # Topology / geometry
    #

    def parse_topology(self):
        """Parse a PDB or standalone XYZ file.

        Returns (mdj_topology, symbols, positions_bohr).
        """
        topology_file_path = self.config.topology_file_path

        if topology_file_path.endswith(".pdb"):
            traj = mdj.load_pdb(topology_file_path)
            topology = traj.topology
            # mdtraj stores positions in nm, convert to Angstrom, then Bohr (atomic units)
            positions = np.asarray(traj.xyz[0], dtype=float) * 10.0 / BOHR
            symbols = [atom.element.symbol for atom in topology.atoms]
        elif topology_file_path.endswith(".xyz"):
            # Don't use mdj.load_xyz since it is only single precision
            with open(topology_file_path, encoding="utf-8") as stream:
                lines = [line.strip() for line in stream if line.strip()]
            try:
                n_atoms = int(lines[0])
            except (IndexError, ValueError) as exc:
                raise ValueError(f"Invalid XYZ atom-count line in {topology_file_path}") from exc
            atom_lines = lines[2 : 2 + n_atoms]
            if len(atom_lines) != n_atoms:
                raise ValueError(f"XYZ declares {n_atoms} atoms but contains {len(atom_lines)} coordinate lines")

            symbols = []
            coordinates_angstrom = []
            for atom_idx, line in enumerate(atom_lines, start=1):
                fields = line.split()
                if len(fields) < 4:
                    raise ValueError(f"Malformed XYZ atom line {atom_idx}: {line}")
                symbols.append(fields[0])
                coordinates_angstrom.append([float(value) for value in fields[1:4]])

            topology = mdj.Topology()
            chain = topology.add_chain()
            residue = topology.add_residue("CPD2", chain)
            for atom_idx, symbol in enumerate(symbols, start=1):
                topology.add_atom(
                    f"{symbol}{atom_idx}",
                    mdj.element.get_by_symbol(symbol),
                    residue,
                )
            positions = np.asarray(coordinates_angstrom, dtype=float) / BOHR
        else:
            raise RuntimeError("Must be a pdb or xyz file.")

        return topology, symbols, positions

    def validate_geometry(self, positions):
        """Require equal competing distances for an unbiased A/B source geometry."""
        config = self.config

        if not config.require_competing_symmetry:
            return

        pairs = config.tracked_boundary_pairs
        competing = [np.linalg.norm(positions[i] - positions[j]) * BOHR for i, j in pairs[1:3]]
        asymmetry = abs(competing[0] - competing[1])
        tolerance = float(config.max_competing_asymmetry_angstrom)
        if asymmetry > tolerance:
            raise ValueError(
                "Starting geometry is not pathway-symmetric: "
                f"competing distances={competing[0]:.10f}, {competing[1]:.10f} A; "
                f"asymmetry={asymmetry:.3e} A > tolerance={tolerance:.3e} A"
            )

        print(
            "Starting pathway symmetry: "
            f"competing distances={competing[0]:.10f}, {competing[1]:.10f} A; "
            f"asymmetry={asymmetry:.3e} A"
        )

    def build_mol(self, symbols, positions):
        config = self.config
        atoms = [(symbol, tuple(coord)) for symbol, coord in zip(symbols, positions, strict=True)]
        mol = pyscf_gto.M(
            atom=atoms,
            basis=config.basis,
            ecp=config.ecp,
            charge=config.charge,
            spin=config.spin,
            unit="Bohr",
        )
        if config.suppress_pyscf_output:
            mol.verbose = 0

        return mol

    #
    # Walker initialization
    #

    def _generate_mb_velocities(self, mol, positions):
        config = self.config
        return (
            pyscf_md.distributions.MaxwellBoltzmannVelocity(mol, config.temperature_kelvin)
            if config.initialize_velocities
            else np.zeros_like(positions)
        )

    def _fit_c2_transform(self, reference, permutation):
        """Return the proper row-vector rotation mapping permuted coordinates to reference."""
        source = reference[permutation]
        source_center = source.mean(axis=0)
        target_center = reference.mean(axis=0)
        u, _, vt = np.linalg.svd((source - source_center).T @ (reference - target_center))
        rotation = u @ vt
        if np.linalg.det(rotation) < 0.0:
            u[:, -1] *= -1.0
            rotation = u @ vt
        residual = np.max(np.linalg.norm((source - source_center) @ rotation + target_center - reference, axis=1))
        if residual * BOHR > 1.0e-4:
            raise ValueError(f"C2 fit residual {residual * BOHR:.3e} A exceeds 1e-4 A")
        return rotation, source_center, target_center

    def _remove_rigid_motion(self, sample_positions, velocity, reference, masses_electron):
        """Recenter positions and remove residual linear/angular velocity."""
        total_mass = masses_electron.sum()
        reference_com = np.einsum("i,ij->j", masses_electron, reference) / total_mass
        sample_com = np.einsum("i,ij->j", masses_electron, sample_positions) / total_mass
        sample_positions = sample_positions + reference_com - sample_com
        velocity = velocity - np.einsum("i,ij->j", masses_electron, velocity) / total_mass
        relative = sample_positions - reference_com
        inertia = np.zeros((3, 3))
        angular_momentum = np.zeros(3)
        for mass, radius, atom_velocity in zip(masses_electron, relative, velocity, strict=True):
            inertia += mass * (np.dot(radius, radius) * np.eye(3) - np.outer(radius, radius))
            angular_momentum += mass * np.cross(radius, atom_velocity)
        angular_velocity = np.linalg.solve(inertia, angular_momentum)
        velocity = velocity - np.cross(angular_velocity, relative)
        return sample_positions, velocity

    def _init_normal_mode_ts_walkers(self, symbols, positions):
        """Classical thermal TS ensemble: n_walkers/2 samples plus their exact C2 partners."""
        config = self.config

        with np.load(config.normal_modes_file_path) as modes:
            mode_symbols = [str(symbol) for symbol in modes["symbols"]]
            reference = np.asarray(modes["coords_A"], dtype=float) / BOHR
            masses_electron = np.asarray(modes["masses_amu"], dtype=float) * AMU_TO_ELECTRON_MASS
            frequencies = np.asarray(modes["freq_au"], dtype=float) / np.sqrt(AMU_TO_ELECTRON_MASS)
            cartesian_modes = np.asarray(modes["norm_mode"], dtype=float)
            permutation = np.asarray(modes["c2_permutation_1based"], dtype=int) - 1

        if mode_symbols != list(symbols):
            raise ValueError("Normal-mode atom symbols/order do not match the starting geometry")
        if not np.allclose(reference, positions, atol=2.0e-5 / BOHR, rtol=0.0):
            raise ValueError("Normal-mode and configured TS coordinates differ by more than 2e-5 A")
        imaginary = np.flatnonzero(frequencies < 0.0)
        if imaginary.size != 1:
            raise ValueError(f"Expected one imaginary mode; found {imaginary.size}")

        # PySCF norm_mode = mass_amu^(-1/2) e. Convert it to e/sqrt(m_e).
        mass_weighted_modes = cartesian_modes * np.sqrt(
            np.asarray(masses_electron / AMU_TO_ELECTRON_MASS)[None, :, None]
        )
        cartesian_modes_au = mass_weighted_modes / np.sqrt(masses_electron)[None, :, None]
        stable = np.flatnonzero(frequencies > 0.0)
        unstable = int(imaginary[0])
        kbt = BOLTZMANN_HARTREE_PER_K * float(config.temperature_kelvin)
        rng = np.random.default_rng(config.initial_ensemble_seed)

        # Orient the imaginary mode toward decreasing mean forming-bond distance.
        unstable_vector = cartesian_modes_au[unstable]
        epsilon = 1.0e-3
        pairs = config.tracked_boundary_pairs

        def mean_forming_distance(coords):
            return np.mean([np.linalg.norm(coords[i] - coords[j]) for i, j in pairs])

        direction = (
            1.0
            if mean_forming_distance(reference + epsilon * unstable_vector)
            < mean_forming_distance(reference - epsilon * unstable_vector)
            else -1.0
        )

        rotation, source_center, target_center = self._fit_c2_transform(reference, permutation)

        samples = []
        for _ in range(config.n_walkers // 2):
            q = rng.normal(size=stable.size) * np.sqrt(kbt) / frequencies[stable]
            qdot = rng.normal(size=stable.size) * np.sqrt(kbt)
            displacement = np.einsum("m,mij->ij", q, cartesian_modes_au[stable])
            velocity = np.einsum("m,mij->ij", qdot, cartesian_modes_au[stable])
            # Positive canonical flux distribution at Q-dagger=0.
            unstable_qdot = np.sqrt(-2.0 * kbt * np.log(rng.uniform()))
            velocity += direction * unstable_qdot * unstable_vector
            sample_positions, velocity = self._remove_rigid_motion(
                reference + displacement, velocity, reference, masses_electron
            )
            samples.append((sample_positions, velocity))

            partner_positions = (sample_positions[permutation] - source_center) @ rotation + target_center
            partner_velocity = velocity[permutation] @ rotation
            samples.append((partner_positions, partner_velocity))

        print(
            f"Initialized {config.n_walkers // 2} classical normal-mode TS samples and "
            f"{config.n_walkers // 2} exact C2 partners (seed={config.initial_ensemble_seed})"
        )
        return samples

    def _init_cartesian_mb_walkers(self, mol, positions):
        config = self.config

        shared_velocity = self._generate_mb_velocities(mol, positions)

        return [
            (
                positions,
                np.copy(shared_velocity)
                if not config.unique_initial_velocities
                else self._generate_mb_velocities(mol, positions),
            )
            for _ in range(config.n_walkers)
        ]

    def make_initial_walkers(self, symbols, positions):
        """Dispatch on config.initialization_mode and return a list of PySCFWalker."""
        config = self.config

        density_kwargs = {}
        if config.density_grid_shape is not None:
            density_kwargs = {"density_grid": np.zeros(config.density_grid_shape)}

        mol = self.build_mol(symbols, positions)

        if config.initialization_mode == "normal_mode_ts":
            initial_phase_points = self._init_normal_mode_ts_walkers(symbols, positions)
        elif config.initialization_mode == "cartesian_mb":
            initial_phase_points = self._init_cartesian_mb_walkers(mol, positions)
        else:
            # Defensive: PySCFSimMakerConfig.__post_init__ already validates this
            raise ValueError(f"Unknown initialization_mode: {config.initialization_mode}")

        weight = 1.0 / config.n_walkers

        return [
            PySCFWalker(
                PySCFState(
                    mol=deepcopy(mol),
                    positions=initial_positions,
                    velocities=initial_velocities,
                    temperature=config.temperature_kelvin,
                    **density_kwargs,
                ),
                weight,
            )
            for initial_positions, initial_velocities in initial_phase_points
        ]

    #
    # Checkpoint / resume / branching
    #

    def _get_next_dir_num(self, base_dir: str) -> int:
        """Get the next directory number given a base directory name."""
        dirs = [d for d in glob(f"{base_dir}_*") if osp.basename(d).rsplit("_", 1)[-1].isdigit()]
        return max(int(osp.basename(d).rsplit("_", 1)[-1]) for d in dirs) + 1 if dirs else 1

    def _get_latest_dir(self, base_dir: str) -> str:
        """Get the latest versioned directory, or the base directory if none exist."""
        dirs = [d for d in glob(f"{base_dir}_*") if osp.basename(d).rsplit("_", 1)[-1].isdigit()]
        if dirs:
            return max(dirs, key=lambda d: int(osp.basename(d).rsplit("_", 1)[-1]))
        return base_dir

    def _get_last_pkl(self, directory: str) -> str:
        """Get the path to the last walker pkl file in a pkls directory."""
        pkls = glob(osp.join(directory, "walkers_cycle_*.pkl"))
        if not pkls:
            raise FileNotFoundError(f"No walker pkl files found in {directory}/")
        return max(pkls, key=lambda p: int(osp.basename(p).removeprefix("walkers_cycle_").removesuffix(".pkl")))

    def resolve_output_directory(self, sub_step, from_branch):
        """Resolve the output directory for a new run, a sub-step, or a branch.

        Returns the resolved output_directory string. Does not create the
        directory on disk.
        """
        config = self.config
        output_directory = config.output_directory

        if sub_step is None:
            if from_branch is not None:
                raise ValueError("from_branch is not supported without a sub_step specified.")

            print("Starting new simulation.")

            # Check base output directory
            if config.write_h5 or config.write_dash or config.store_pickles:
                if osp.isdir(output_directory):
                    if config.overwrite:
                        print(f"Warning: output directory already exists, overwriting: {output_directory}/")
                    else:
                        # Create new versioned directory if not overwriting
                        next_dir_num = self._get_next_dir_num(output_directory)
                        output_directory = f"{output_directory}_{next_dir_num}"
                        print(f"Warning: output directory already exists, creating new directory: {output_directory}/")
                else:
                    print(f"Creating output directory: {output_directory}/")

            return output_directory

        # Sub-step provided: resolve to the latest versioned base directory
        output_directory = self._get_latest_dir(output_directory)

        if sub_step == 0:
            if from_branch is not None:
                raise ValueError("from_branch is not supported with sub_step 0.")
            print("Starting simulation in sub-step mode.")
        else:
            # Base directory must already exist in sub-step mode
            if not osp.isdir(output_directory):
                raise FileNotFoundError(f"Output directory does not exist: {output_directory}/")
            print(f"Continuing simulation at sub-step {sub_step}.")

        # Get next sub-step directory
        sub_directory = f"sub_{sub_step}"
        if from_branch is not None:
            sub_directory += f"_branch_{from_branch}"

        # Output directory already exists, resolve versioned name if not overwriting
        if osp.isdir(osp.join(output_directory, sub_directory)):
            if config.overwrite:
                output_directory = osp.join(output_directory, sub_directory)
                print(f"Warning: sub-step output directory already exists, overwriting: {output_directory}/")
            else:
                if from_branch is not None:
                    raise ValueError(
                        f"Sub-step directory already exists and overwrite is disabled: "
                        f"{osp.join(output_directory, sub_directory)}"
                    )

                # Create new versioned branch directory if not overwriting
                next_branch_num = self._get_next_dir_num(osp.join(output_directory, f"sub_{sub_step}_branch"))
                output_directory = osp.join(output_directory, f"sub_{sub_step}_branch_{next_branch_num}")
                print(f"Warning: sub-step output directory already exists, creating new directory: {output_directory}/")
        else:
            output_directory = osp.join(output_directory, sub_directory)
            print(f"Creating sub-step output directory: {output_directory}/")

        return output_directory

    def resolve_walkers(self, output_directory, sub_step, from_branch, source_walkers):
        """Return (walkers, start_cycle) for a new run, sub-step, or branch."""
        if sub_step is None or sub_step == 0:
            return source_walkers, None

        # If sub-step provided, resolve to the base output directory
        base_output_directory = self._get_latest_dir(self.config.output_directory)

        # Get previous sub-step directory
        prev_sub_directory = f"sub_{sub_step - 1}"
        if from_branch is not None:
            prev_sub_directory += f"_branch_{from_branch}"
        prev_pkls_directory = osp.join(base_output_directory, prev_sub_directory, "pkls")

        # Load walkers from the source directory
        target_pkl = self._get_last_pkl(prev_pkls_directory)
        print(f"Resuming from: {target_pkl}")

        # 0-based cycle index in the filename, so add 2 for the next cycle
        start_cycle = int(target_pkl.rsplit("_", 1)[-1].removesuffix(".pkl")) + 2

        # Restore walkers from the pkl
        with open(target_pkl, "rb") as f:
            walkers = pickle.load(f)  # noqa: S301

        return walkers, start_cycle

    #
    # Apparatus construction
    #

    def make_runner(self):
        config = self.config
        return PySCFRunner(
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

    def make_resampler(self, init_state):
        config = self.config

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

    def make_bc(self, walkers, source_walkers):
        config = self.config

        if not config.use_boundary_conditions:
            return NoBC()

        # FIXME: Is this used anymore?
        # return PySCFBondDistanceBC(
        #     initial_states=[walker.state for walker in walkers],
        #     break_pairs=config.break_pairs,
        #     break_cutoffs=config.break_cutoffs,
        #     make_pairs=config.make_pairs,
        #     make_cutoffs=config.make_cutoffs,
        # )
        return PySCFMultiBoundaryBC(
            n_walkers=len(walkers),
            # TODO: deepcopy needed here?
            initial_states=[deepcopy(walker.state) for walker in source_walkers],
            initial_weights=[walker.weight for walker in source_walkers],
            boundary_definitions=config.boundary_definitions,
            tracked_pairs=config.tracked_boundary_pairs,
        )

    def make_apparatus(self, walkers, source_walkers):
        """Return (runner, resampler, boundary_conditions)."""
        runner = self.make_runner()
        resampler = self.make_resampler(walkers[0].state)
        boundary_conditions = self.make_bc(walkers, source_walkers)

        return runner, resampler, boundary_conditions

    #
    # Work mapper
    #

    def _make_cpu_mapper(self):
        return PySCFCPUWorkerMapper(num_workers=self.config.n_walkers)

    def _resolve_gpu_device_ids(self):
        config = self.config

        cuda_visible_devices = config._cuda_visible_devices_env_var
        if cuda_visible_devices == "":
            raise RuntimeError("No GPUs available: CUDA_VISIBLE_DEVICES is not set or empty.")

        available_ids = [int(x) for x in cuda_visible_devices.split(",")]
        num_available = config._num_gpus_visible
        print(f"Found {num_available} available devices.")

        # Round-robin assign workers to GPUs
        return [available_ids[i % num_available] for i in range(config.n_walkers)]

    def _make_gpu_mapper(self):
        if importlib.util.find_spec("cupy") is None:
            raise SystemExit(
                "GPU backend requested but CuPy is not installed. "
                "Install a CUDA-matched CuPy package (e.g. cupy-cuda12x) "
                "or rerun with CPU.",
            )

        device_ids = self._resolve_gpu_device_ids()

        return PySCFGPUWorkerMapper(num_workers=self.config.n_walkers, platform="CUDA", device_ids=device_ids)

    def make_work_mapper(self):
        config = self.config

        if config.backend == "CPU":
            return self._make_cpu_mapper()
        elif config.backend == "GPU":
            return self._make_gpu_mapper()

        raise ValueError(f"Unknown backend: {config.backend}")

    #
    # Reporters
    #

    def _make_pkl_reporter(self, output_directory, start_cycle):
        return WalkerPklReporter(
            save_dir=osp.join(output_directory, "pkls"),
            freq=1,
            num_backups=2,
            start_cycle=start_cycle,
        )

    def _make_hdf5_reporter(self, output_directory, output_mode, json_top, runner, resampler, boundary_conditions):
        config = self.config

        h5_save_fields = PySCFHDF5Reporter.DEFAULT_SAVE_FIELDS
        if config.density_grid_shape is not None:
            # TODO: Omit `density_matrix` by default because its array shape depends
            # on the AO basis size and can be expensive to store.
            h5_save_fields += (
                # "density_matrix",
                "density_grid",
                "density_grid_origin",
                "density_grid_spacing",
            )

        return PySCFHDF5Reporter(
            save_fields=h5_save_fields,
            file_path=config.get_h5_path(output_directory),
            mode=output_mode,
            topology=json_top,
            resampler=resampler,
            boundary_conditions=boundary_conditions,
        )

    def _make_dashboard_reporter(self, output_directory, output_mode, runner):
        config = self.config

        return DashboardReporter(
            file_path=config.get_dash_path(output_directory),
            mode=output_mode,
            runner_dash=PySCFRunnerDashboardSection(runner),
        )

    def make_reporters(self, output_directory, runner, resampler, boundary_conditions, json_top, start_cycle):
        config = self.config
        output_mode = "w" if config.overwrite else "x"
        reporters = []

        if config.store_pickles:
            reporters.append(self._make_pkl_reporter(output_directory, start_cycle))

        if config.write_h5:
            reporters.append(
                self._make_hdf5_reporter(
                    output_directory, output_mode, json_top, runner, resampler, boundary_conditions
                )
            )

        if config.write_dash:
            reporters.append(self._make_dashboard_reporter(output_directory, output_mode, runner))

        return reporters

    #
    # Sim manager assembly
    #

    def make_sim_manager(self, sub_step=None, from_branch=None):
        config = self.config

        mdj_top, symbols, positions = self.parse_topology()
        self.validate_geometry(positions)

        # Fixed reactant source ensemble for product recycling
        # Remains the source even when continuing from a sub-step
        source_walkers = self.make_initial_walkers(symbols, positions)

        output_directory = self.resolve_output_directory(sub_step, from_branch)
        walkers, start_cycle = self.resolve_walkers(output_directory, sub_step, from_branch, source_walkers)

        if not osp.exists(output_directory):
            os.makedirs(output_directory)

        mapper = self.make_work_mapper()
        runner, resampler, boundary_conditions = self.make_apparatus(walkers, source_walkers)
        json_top = mdtraj_to_json_topology(mdj_top)

        reporters = self.make_reporters(output_directory, runner, resampler, boundary_conditions, json_top, start_cycle)

        sim_manager = Manager(
            walkers,
            runner=runner,
            work_mapper=mapper,
            resampler=resampler,
            boundary_conditions=boundary_conditions,
            reporters=reporters,
        )

        return sim_manager

    #
    # Top-level entry point
    #

    def run(self, sub_step=None, from_branch=None):
        config = self.config

        sim_manager = self.make_sim_manager(sub_step=sub_step, from_branch=from_branch)

        start_time = perf_counter()
        end_walkers, _ = sim_manager.run_simulation(
            n_cycles=config.n_cycles,
            segment_lengths=config.segment_length,
        )
        total_time = perf_counter() - start_time

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
            f"Scanner caching: {config.use_scanner_caching}, "
            # TODO: Allow passing in keys and arguments that you want printed
            # f"Multi BC: {config.use_boundary_conditions}; "
            # f"product cutoff={config.product_cutoff_angstrom:g} A; "
            # f"reactant cutoff={config.reactant_cutoff_angstrom:g} A",
        )

        return end_walkers
