# Running Wepy-PySCF

## Startup Script

To run simulations on the GPU, `gpu4pyscf` must be in your path. cuTENSOR must also be in your path to avoid falling back to CuPy during GPU simulations. The following script makes this easy to do and loads the required modules. Copy and paste the following script below somewhere convinient (like `~/load_wepy_gpu.sh`).

```bash
name=your_name
export PYTHONPATH="/mnt/research/PTR_bose/$name/gpu4pyscf${PYTHONPATH:+:$PYTHONPATH}"
export LD_LIBRARY_PATH=/mnt/home/"$USER"/.conda/envs/wepy-dev/lib/python3.10/site-packages/cutensor/lib:$LD_LIBRARY_PATH
ml purge && ml load Miniforge3 OpenBLAS CUDA && conda activate wepy-dev
cd /mnt/research/PTR_bose/$name/wepy_dev
```

Then when you ssh into a dev node or in your SLURM script, run:

```bash
source load_wepy_gpu.sh
```

This will set up your session to be able to run the simulations.

## Running

If running on a GPU, you will need to set the CUDA visible devices environment variable to the indicies of your GPUs.

```bash
# For 2 GPUs
export CUDA_VISIBLE_DEVICES=0,1

# For 4 GPUs
export CUDA_VISIBLE_DEVICES=0,1,2,3
```

Then you can run the example script with:

```bash
python info/examples/PySCF/alanine.py
```

Note that the output folder will be written to the directory you ran your system from.

## Modifying Parameters

To modify the parameters of the simulation, you will need to edit the `info/examples/PySCF/alanine.py` file, which contains many different options. You can also create your own file to run a simulation for a different system. Use the provided examples for reference. These files define the `PySCFInput` class to define the simulation parameters and run the wepy simulation with the `run()` method from `revo_pyscf.py`.

## Sub-Steps and Branching

The sub-step and branching features allows you to resume a simulation from a previous checkpoint and explore multiple trajectories from the same starting point. Branches are automatically created when a sub-step is ran multiple times to avoid overwriting previous results. Two optional parameters are available:

- `--sub-step SUB_STEP` - Enables sub-step mode, organizing outputs into numbered sub-directories (sub_0, sub_1, ...) so a simulation can be paused and resumed across multiple runs. Must be set to 0 on the first run to enable sub-step mode for future continuation; omitting it doesn't create the required sub-step directories.
- `--from-branch FROM_BRANCH` - Requires `--sub-step`. Specifies which branch of the previous sub-step to resume from. Defaults to the latest branch. For example, running `--sub-step 1 --from-branch 2` resumes from `sub_0_branch_2` and writes output to `sub_1_branch_2`. Useful for exploring different trajectories from the same checkpoint without overwriting previous results.

### Example

Branching relies on walker pickle files, that maintain a continuous cycle index across sub-steps. Each continuation picks up exactly where the previous left off. By default, the most recent two pickles are saved for each cycle. This allows easy continuation in the event of a simulation crash or interruption. The example below shows a simulation run without and with sub-step mode and branching (10 cycles each run):

```
├── alanine_4W_3C_1S_VelocityVerlet    # Run without --sub-step
│   └── pkls
│       ├── walkers_cycle_8.pkl
│       └── walkers_cycle_9.pkl
└── alanine_4W_3C_1S_VelocityVerlet_1  # Started with --sub-step 0
    ├── sub_0
    │   └── pkls
    │       ├── walkers_cycle_8.pkl
    │       └── walkers_cycle_9.pkl
    ├── sub_1                          # Continued with --sub-step 1
    │   └── pkls
    │       ├── walkers_cycle_18.pkl
    │       └── walkers_cycle_19.pkl
    ├── sub_2                          # Continued with --sub-step 2
    │   └── pkls
    │       ├── walkers_cycle_28.pkl
    │       └── walkers_cycle_29.pkl
    ├── sub_2_branch_1                 # --sub-step 2 (branch created)
    │   └── pkls
    │       ├── walkers_cycle_28.pkl
    │       └── walkers_cycle_29.pkl
    ├── sub_2_branch_2                 # --sub-step 2 (branch created)
    │   └── pkls
    │       ├── walkers_cycle_28.pkl
    │       └── walkers_cycle_29.pkl
    ├── sub_3                          # Continued with --sub-step 3
    │   └── pkls
    │       ├── walkers_cycle_38.pkl
    │       └── walkers_cycle_39.pkl
    └── sub_3_branch_2                 # --sub-step 3 --from-branch 2
        └── pkls
            ├── walkers_cycle_38.pkl
            └── walkers_cycle_39.pkl
```
