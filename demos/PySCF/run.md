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

Then you can run the script with:

```bash
python info/examples/PySCF/alanine.py
```

Note that output files will be written to the directory you ran `revo_pyscf.py` from.

## Modifying Parameters

To modify the parameters of the simulation, you will need to edit the `info/examples/PySCF/alanine.py` file, which contains many different options. You can also create your own file to run a simulation for a different system. Use the provided examples for reference. These files use the `PySCFInput` class from `pyscf_input.py` to define the simulation parameters and run the simulation with the `run()` method from `revo_pyscf.py`.
