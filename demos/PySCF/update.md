# Updating Wepy Development Environments

**MAKE SURE YOU ARE IN THE CONDA ENVIRONMENT BEFORE FOLLOWING THESE STEPS.**

## Updating Wepy Dev

Run the following commands to update to the latest version:

```bash
cd wepy_dev
git pull
ml purge && ml load Miniforge3 OpenBLAS CUDA
make build && pip uninstall wepy -y && pip install dist/wepy-1.1.0-py2.py3-none-any.whl
```

## Updating PySCF Dev

Run the following commands to update to the latest version:

```bash
cd pyscf_dev
git pull
./conda/build.sh
```
