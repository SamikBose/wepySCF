"""Set up a REVO simulation with PySCF dynamics for alanine dipeptide.

This version uses a separate `pyscf_input.py` file for all PySCF/simulation parameters.
"""

# Set the default number of threads before importing libraries
import os

os.environ.setdefault("OMP_NUM_THREADS", "1")  # Good default for PySCF CPU runs, but can be overridden by the user

# Standard Library
import importlib.util
from time import perf_counter

# Third Party Library
import mdtraj as mdj
import numpy as np
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

    return [
        PySCFWalker(
            PySCFState(
                symbols=symbols,
                positions=positions,
                charge=0,
                spin=0,
                velocities=np.zeros_like(positions),
                # TODO: Initialize with Maxwell-Boltzmann
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


def main():
    # build_mapper
    if CONFIG.backend == "gpu":
        if importlib.util.find_spec("cupy") is None:
            raise SystemExit(
                "GPU backend requested but CuPy is not installed. "
                "Install a CUDA-matched CuPy package (e.g. cupy-cuda12x) "
                "or rerun with CPU.",
            )

        cuda_visible_devices = os.environ.get("CUDA_VISIBLE_DEVICES", None)
        if cuda_visible_devices is not None:
            num_available = len([x for x in cuda_visible_devices.split(",") if x.strip()])
            print(f"Found {num_available} available devices.")
        else:
            raise RuntimeError("No GPUs available: CUDA_VISIBLE_DEVICES is not set or empty.")

        if num_available == 0:
            raise RuntimeError("No GPUs found.")

        device_ids = [i % num_available for i in range(CONFIG.n_walkers)]  # Round-robin assign workers to GPUs
        mapper = PySCFGPUWorkerMapper(num_workers=CONFIG.n_walkers, platform="CUDA", device_ids=device_ids)

    elif CONFIG.backend == "cpu":
        mapper = PySCFCPUWorkerMapper(num_workers=CONFIG.n_walkers)

    mdj_top, symbols, positions = parse_with_mdtraj_topology(CONFIG.topology_file_path)

    walkers = generate_initial_walkers(
        symbols=symbols,
        positions=positions,
        n_walkers=CONFIG.n_walkers,
        density_grid_shape=CONFIG.density_grid_shape,
    )

    runner = PySCFRunner(
        basis=CONFIG.basis,
        method=CONFIG.method,
        xc=CONFIG.xc,
        dt=CONFIG.dt,
        integrator_cls=CONFIG.integrator_cls,
        integrator_kwargs=CONFIG.integrator_kwargs,
        temperature_kelvin=CONFIG.temperature_kelvin,
        backend=CONFIG.backend,
        density_grid_shape=CONFIG.density_grid_shape,
    )

    resampler = build_revo_resampler(walkers[0].state)

    json_topology = mdtraj_to_json_topology(mdj_top)
    output_mode = "w" if CONFIG.overwrite else "x"

    reporters = []

    h5_save_fields = PySCFHDF5Reporter.DEFAULT_SAVE_FIELDS
    if CONFIG.density_grid_shape is not None:
        # We omit `density_matrix` by default because its array shape depends on
        # the AO basis size and can be expensive to store. Could store this later.
        h5_save_fields += (
            # "density_matrix",
            "density_grid",
            "density_grid_origin",
            "density_grid_spacing",
        )
    if CONFIG.write_h5:
        h5_reporter = PySCFHDF5Reporter(
            save_fields=h5_save_fields,
            file_paths=[CONFIG.h5_path],
            modes=[output_mode],
            topology=json_topology,
            resampler=resampler,
            boundary_conditions=NoBC(),
        )
        reporters.append(h5_reporter)

    if CONFIG.write_dash:
        dash_reporter = DashboardReporter(
            file_paths=[CONFIG.dash_path],
            modes=[output_mode],
            runner_dash=PySCFRunnerDashboardSection(runner=runner),
        )
        reporters.append(dash_reporter)

    sim_manager = Manager(
        walkers,
        runner=runner,
        work_mapper=mapper,
        resampler=resampler,
        boundary_conditions=NoBC(),
        reporters=reporters,
    )

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
        f"System: {CONFIG.system.title()}, Basis: {CONFIG.basis}, Method: {CONFIG.method}"
        + (f"/{CONFIG.xc}" if CONFIG.xc else ""),
    )
    if CONFIG.backend == "gpu":
        print(f"GPU device IDs: {device_ids}")
    elif CONFIG.backend == "cpu":
        print(f"CPU workers: {CONFIG.n_walkers}")
    print(f"OpenMP threads: {CONFIG._omp_threads_env_var}")  # noqa: SLF001
    temperatures = [walker.state.get("temperature").item() for walker in end_walkers]
    potentials = [walker.state.get("potential").item() for walker in end_walkers]
    kinetics = [walker.state.get("kinetic").item() for walker in end_walkers]
    energies = [p + k for p, k in zip(potentials, kinetics, strict=True)]
    print("Final walker temperatures:", temperatures)
    print("Final walker energies:", energies)
    print("Final walker potentials:", potentials)
    print("Final walker kinetics:", kinetics)


if __name__ == "__main__":
    main()
