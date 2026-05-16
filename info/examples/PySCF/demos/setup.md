# Wepy Development Environments

Below are steps for setting up development environments for Wepy on the HPCC and locally.

## HPCC Development

### Conda Setup

![Conda Setup](conda.gif)

1. Join a dev node on the HPCC.

```bash
# CPU only
ssh dev-amd24

# CPU and GPU
ssh dev-amd24-h200
```

2. Enter your directory on the HPCC.

```bash
cd /mnt/research/PTR_bose/your_name
```

3. Load the required modules.

```bash
ml purge && ml load Miniforge3 OpenBLAS CUDA
```

4. Create and activate the Conda environment from the `environment_minimal.yml` template.

```bash
conda env create -f ../environment_minimal.yml
conda activate wepy-dev
```

### Wepy Dev Setup

![Wepy Dev Setup](wepy_dev.gif)

1. Make sure to be on a dev node, have the required modules loaded, and have the Conda environment activated.

```bash
ssh dev-amd24-h200
ml purge && ml load Miniforge3 OpenBLAS CUDA
conda activate wepy-dev
```

2. Clone the `wepy_dev` repository and enter the directory.

```bash
git clone https://github.com/SamikBose/wepy_dev.git
cd wepy_dev
```

3. Build the Wepy package using `make`.

```bash
make build
```

4. Remove any existing Wepy packages.

```bash
pip uninstall wepy -y
```

5. Install the newly built Wepy package.

```bash
pip install dist/wepy-1.1.0-py2.py3-none-any.whl
```

### PySCF Dev Setup

![PySCF Dev Setup](pyscf_dev.gif)

1. Make sure to be on a dev node, have the required modules loaded, and have the Conda environment activated.

```bash
ssh dev-amd24-h200
ml purge && ml load Miniforge3 OpenBLAS CUDA
conda activate wepy-dev
```

2. Clone the `pyscf_dev` repository and enter the directory.

```bash
git clone https://github.com/SamikBose/pyscf_dev.git
cd pyscf_dev
```

3. Build the PySCF package using the Conda build script.

```bash
./conda/build.sh
```

### PySCF GPU Setup

![PySCF GPU Setup](gpu4pyscf.gif)

1. Make sure to be on a dev node with a GPU, have the required modules loaded, and have the Conda environment activated.

```bash
ssh dev-amd24-h200
ml purge && ml load Miniforge3 OpenBLAS CUDA
conda activate wepy-dev
```

2. Install the appropriate packages based on your CUDA version.

```bash
# Check your CUDA version
nvcc --version

# For CUDA 12.x
pip3 install cutensor-cu12 cupy-cuda12x
```

3. Clone the [`gpu4pyscf`](https://github.com/pyscf/gpu4pyscf) repository and enter the directory.

```bash
git clone https://github.com/pyscf/gpu4pyscf.git
cd gpu4pyscf
```

4. Build the PySCF GPU package.

```bash
ml load CMake # Ensure CMake is loaded
cmake -S gpu4pyscf/lib -B build/temp.gpu4pyscf
cmake --build build/temp.gpu4pyscf -j 4 # This will take a long time
```

5. Add the `gpu4pyscf` library to your `PYTHONPATH` environment variable.

```bash
CURRENT_PATH=`pwd`
export PYTHONPATH="${PYTHONPATH}:${CURRENT_PATH}"
```

6. Navigate back to `wepy_dev` and run your Wepy simulations with GPU support.

```bash
cd ../wepy_dev
python python info/examples/PySCF/source/revo_pyscf_alanine.py
```

7. You might have to reinstall `mdtraj` if you get `numpy`-related error.

```bash
pip install --upgrade --force-reinstall mdtraj --no-cache-dir
```

## Local Development

![Nix + Pixi Setup](nix_pixi_cpu.gif)

1. Install [Nix](https://nixos.org/) and [Pixi](https://pixi.prefix.dev/latest/) on your local machine.

2. Clone the `wepy_dev` repository and enter the directory.

```bash
git clone https://github.com/SamikBose/wepy_dev.git
cd wepy_dev
```

3. Enter the Nix development shell.

```bash
nix develop
```

4. Enter the Pixi development shell (will automatically install dependencies).

```bash
pixi shell
```

5. Build the Wepy package using `make`.

```bash
make build
```

6. Install the newly built Wepy package.

```bash
uv pip install dist/wepy-1.1.0-py2.py3-none-any.whl
```
