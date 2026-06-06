# Make sure your base `gpu4pyscf` folder is not called `gpu4pyscf` to avoid confusion with the cloned repository

Require zoxide

Output gpu4pyscf.gif

Set Width 1000 # TODO: Increase width to prevent wrapping?
Set Height 600
Set Padding 0
Set Margin 35

Set Theme "Catppuccin Mocha"
Set MarginFill "#1e1e2e" # Set margin background to theme color
# Set FontFamily "Monaspace Neon Frozen" # Not working on HPCC
Set FontSize 14

# Initialization
Set Shell "bash"
Hide
Type "source ~/.bashrc"
Enter
Type "cd nathan"
Enter
Type "ml purge && ml load Miniforge3 OpenBLAS CUDA"
Enter
Type "conda activate wepy-dev"
Enter
Type "clear"
Enter
Wait /.+@.+:.*\$/ # Wait for the prompt to return (default cap of 15s)
Show


# Check your CUDA version
Sleep 1s
Type "nvcc --version"
Enter
Sleep 1s

# Install the appropriate CuPy and cuTENSOR packages based on your CUDA version
Type "pip install cutensor-cu12 cupy-cuda12x"
Enter
Sleep 5s

Hide
Sleep 15s # Skip 15s ahead
Show

Wait@60s /.+@.+:.*\$/ # Wait for the prompt to return (hard cap of 60s)
Sleep 1s

# Clone the gpu4pyscf repository and enter it
Type "git clone https://github.com/pyscf/gpu4pyscf.git"
Enter
Wait /.+@.+:.*\$/ # Wait for the prompt to return
Sleep 1s
Type "cd gpu4pyscf"
Enter
Sleep 1s

# Build the project
Type "ml load CMake"
Enter
Wait /.+@.+:.*\$/ # Wait for the prompt to return
Sleep 1s
Type "cmake -S gpu4pyscf/lib -B build/temp.gpu4pyscf"
Enter
Wait@30s /.+@.+:.*\$/ # Wait for the prompt to return
Sleep 1s
Type "cmake --build build/temp.gpu4pyscf -j 4"
Enter
Sleep 5s

Hide # Skip to the end of the build process
Wait@1800s /.+@.+:.*\$/ # Wait for the prompt to return (hard cap of 30m)
Show

Sleep 2s

# Make sure it is in your path
Type "CURRENT_PATH=`pwd`"
Enter
Sleep 1s
Type `export PYTHONPATH="${PYTHONPATH}:${CURRENT_PATH}"`
Enter

Sleep 3s
