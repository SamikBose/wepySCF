"""Set up a REVO simulation with PySCF dynamics for alanine dipeptide.

This version uses a separate `pyscf_input.py` file for all PySCF/simulation parameters.
"""

# Set the default number of threads before importing libraries
import os

os.environ.setdefault("OMP_NUM_THREADS", "1")  # Good default for PySCF CPU runs, but can be overridden by the user

# Standard Library
import importlib.util
import subprocess
import tempfile
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
from wepy.resampling.distances.pyscf import QMGridDensityDistance
from wepy.resampling.resamplers.revo import REVOResampler
from wepy.runners.pyscf import PySCFCPUWorkerMapper, PySCFGPUWorkerMapper, PySCFRunner, PySCFState, PySCFWalker
from wepy.sim_manager import Manager
from wepy.util.mdtraj import mdtraj_to_json_topology

ALANINE_DIPEPTIDE_PDB = """\
ATOM      1 1HH3 ACE     1       2.000   1.000  -0.000
ATOM      2  CH3 ACE     1       2.000   2.090   0.000
ATOM      3 2HH3 ACE     1       1.486   2.454   0.890
ATOM      4 3HH3 ACE     1       1.486   2.454  -0.890
ATOM      5  C   ACE     1       3.427   2.641  -0.000
ATOM      6  O   ACE     1       4.391   1.877  -0.000
ATOM      7  N   ALA     2       3.555   3.970  -0.000
ATOM      8  H   ALA     2       2.733   4.556  -0.000
ATOM      9  CA  ALA     2       4.853   4.614  -0.000
ATOM     10  HA  ALA     2       5.408   4.316   0.890
ATOM     11  CB  ALA     2       5.661   4.221  -1.232
ATOM     12 1HB  ALA     2       5.123   4.521  -2.131
ATOM     13 2HB  ALA     2       6.630   4.719  -1.206
ATOM     14 3HB  ALA     2       5.809   3.141  -1.241
ATOM     15  C   ALA     2       4.713   6.129   0.000
ATOM     16  O   ALA     2       3.601   6.653   0.000
ATOM     17  N   NME     3       5.846   6.835   0.000
ATOM     18  H   NME     3       6.737   6.359  -0.000
ATOM     19  CH3 NME     3       5.846   8.284   0.000
ATOM     20 1HH3 NME     3       4.819   8.648   0.000
ATOM     21 2HH3 NME     3       6.360   8.648   0.890
ATOM     22 3HH3 NME     3       6.360   8.648  -0.890
END
"""


def parse_with_mdtraj_topology(pdb_text):
    with tempfile.NamedTemporaryFile(suffix=".pdb", mode="w", delete=True) as tmp:
        tmp.write(pdb_text)
        tmp.flush()
        traj = mdj.load_pdb(tmp.name)

    topology = traj.topology
    # mdtraj stores positions in nm, convert to Angstrom, then Bohr (atomic units)
    positions = np.asarray(traj.xyz[0], dtype=float) * 10.0 / BOHR
    symbols = [atom.element.symbol for atom in topology.atoms]

    return topology, symbols, positions


def generate_initial_walkers(symbols, positions, n_walkers, density_grid_shape):
    weight = 1.0 / n_walkers

    return [
        PySCFWalker(
            PySCFState(
                symbols=symbols,
                positions=positions,
                charge=0,
                spin=0,
                velocities=np.zeros_like(positions),
                accelerations=None,
                # Store as 1D feature arrays so the HDF5 reporter can extend them
                potential=np.array([np.nan], dtype=float),
                kinetic=np.array([np.nan], dtype=float),
                density_matrix=np.zeros((len(symbols), len(symbols))),  # TODO: Is this right?
                density_grid=np.zeros(density_grid_shape),
                density_grid_origin=np.zeros(3),
                density_grid_spacing=np.ones(3),
            ),
            weight,
        )
        for _ in range(n_walkers)
    ]


def build_revo_resampler(init_state):
    distance = QMGridDensityDistance(grid_key="density_grid", normalize=True)

    return REVOResampler(
        distance=distance,
        init_state=init_state,
        merge_dist=0.5,
        char_dist=1.0,
        pmin=1e-12,
        pmax=0.99,
    )


def main():
    if CONFIG.backend == "gpu":
        if importlib.util.find_spec("cupy") is None:
            raise SystemExit(
                "GPU backend requested but CuPy is not installed. "
                "Install a CUDA-matched CuPy package (e.g. cupy-cuda12x) "
                "or rerun with CPU.",
            )
        # Get number of GPUs using nvidia-smi
        try:
            result = subprocess.run(
                ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
                capture_output=True,
                text=True,
                check=True,
            )
            num_gpus = len([line for line in result.stdout.strip().split("\n") if line])

            if num_gpus == 0:
                raise RuntimeError("No GPUs found.")

            print(f"Found {num_gpus} GPU(s) available for PySCFRunner.")

            num_workers = CONFIG.num_workers or CONFIG.n_walkers
            device_ids = [i % num_gpus for i in range(num_workers)]  # Round-robin assign workers to GPUs
            mapper = PySCFGPUWorkerMapper(num_workers=num_workers, platform="CUDA", device_ids=device_ids)
        except (FileNotFoundError, subprocess.CalledProcessError):
            raise RuntimeError("No GPUs found or nvidia-smi failed.") from None

    elif CONFIG.backend == "cpu":
        num_workers = CONFIG.num_workers or CONFIG.n_walkers
        mapper = PySCFCPUWorkerMapper(num_workers=num_workers)

    mdj_top, symbols, positions = parse_with_mdtraj_topology(ALANINE_DIPEPTIDE_PDB)

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

    resampler = build_revo_resampler(init_state=walkers[0].state)

    json_topology = mdtraj_to_json_topology(mdj_top)
    output_mode = "w" if CONFIG.overwrite else "x"

    reporters = []

    if CONFIG.write_h5:
        h5_reporter = PySCFHDF5Reporter(
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
        f"({total_time / CONFIG.n_cycles:.3f} sec / cycle)"
    )
    print(
        f"{len(end_walkers)} walkers, {CONFIG.n_cycles} cycles * {CONFIG.segment_length} steps "
        f"({CONFIG.n_cycles * CONFIG.segment_length} total MD steps)"
    )
    print(f"Basis: {CONFIG.basis}, Method: {CONFIG.method}" + (f"/{CONFIG.xc}" if CONFIG.xc else ""))
    if CONFIG.backend == "gpu":
        print(f"GPU device IDs: {device_ids}")
    elif CONFIG.backend == "cpu":
        print(f"CPU workers: {num_workers}")
    print(f"OpenMP threads: {CONFIG._omp_threads_env_var}")  # noqa: SLF001
    # print("Final walker potentials:", [walker.state.get("potential") for walker in end_walkers]) # Less precision
    print("Final walker potentials:", [walker.state.get("potential").item() for walker in end_walkers])
    # print("Final walker kinetics:", [walker.state.get("kinetic").item() for walker in end_walkers])


if __name__ == "__main__":
    main()
