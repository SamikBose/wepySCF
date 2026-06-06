Require zoxide

Output conda.gif

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
Type "cd"
Enter
Env HOSTNAME "gateway-01" # Simulate starting on gateway node
Enter
Type "clear"
Enter
Wait /.+@.+:.*\$/ # Wait for the prompt to return (default cap of 15s)
Show


# SSH into a dev node
Sleep 1s
Type "ssh dev-amd24-h200"
Enter
Wait /.+@.+:.*\$/ # Wait for the prompt to return
Sleep 1s
Type "clear"
Enter
Sleep 1s

# Enter the PTR_bose directory
Type "cd /mnt/research/PTR_bose"

Hide # Shorten the path to avoid future commands from being too long (/mnt/research/PTR_bose -> ~/PTR_bose)
Enter
Type "cd"
Enter
Type "cd PTR_bose"
Enter
Type "clear"
Enter
Wait /.+@.+:.*\$/ # Wait for the prompt to return
Show # Back to our regularly scheduled programming

Sleep 1s

# Enter your directory
Type "cd nathan"
Enter
Sleep 1s

# Purge current loaded modules and load the required ones
Type "ml purge && ml load Miniforge3 OpenBLAS CUDA"
Enter
Wait /.+@.+:.*\$/ # Wait for the prompt to return
Sleep 1s

# Create the Conda environment from the minimal template
Type "conda env create -f ../environment_minimal.yml"
Enter
Sleep 3s
Hide
Wait@600s /.+@.+:.*\$/ # Wait for the prompt to return (hard cap of 10m)
Show
Sleep 1.5s

# Activate the environment
Type "conda activate wepy-dev"
Enter

Sleep 3s
