# Make sure your base `pyscf_dev` folder is not called `pyscf_dev` to avoid confusion with the cloned repository

Output pyscf_dev.gif

Set Width 1000
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


# Clone the repository
Sleep 1s
Type "git clone https://github.com/SamikBose/pyscf_dev.git"
Enter
Wait /.+@.+:.*\$/ # Wait for the prompt to return
Sleep 1s

# Enter the directory
Type "cd pyscf_dev"
Enter
Sleep 1s


# Build the project (make already installed from the Conda environment)
Type "./conda/build.sh"
Enter
Sleep 3s

Hide
Wait@3600s /.+@.+:.*\$/ # Wait for the prompt to return (hard cap of 1h)
Show

Sleep 3s
