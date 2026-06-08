# Make sure your base `wepy_dev` folder is not called `wepy_dev` to avoid confusion with the cloned repository

Require zoxide

Output wepy_dev.gif

Set Width 1100
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
Type "git clone https://github.com/SamikBose/wepy_dev.git"
Enter
Wait /.+@.+:.*\$/ # Wait for the prompt to return
Sleep 1s

# Enter the directory
Type "cd wepy_dev"
Enter
Sleep 1s

# Build the project (make already installed from the Conda environment)
Type "make build"
Enter
Sleep 3s

Hide
Sleep 7s # Skip 7s ahead
Show

Wait@60s /.+@.+:.*\$/ # Wait for the prompt to return (hard cap of 60s)
Sleep 1s

# Remove the existing package (don't use `pip install --force-reinstall` since that takes a lot longer)
Type "pip uninstall wepy -y"
Enter
Wait /.+@.+:.*\$/ # Wait for the prompt to return
Sleep 1s

Type "pip install dist/wepy-1.1.0-py2.py3-none-any.whl"
Enter
Wait@120s /.+@.+:.*\$/ # Wait for the prompt to return

Sleep 3s
