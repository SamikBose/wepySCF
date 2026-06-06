Require zoxide

Output nix_pixi_cpu.gif

Set Width 1000
Set Height 600
Set Padding 0
Set Margin 35

Set Theme "Catppuccin Mocha"
Set FontSize 14
Set FontFamily "Monaspace Neon Frozen"
Set MarginFill "#1e1e2e" # Set margin background to theme color


Set Shell "bash"
Hide
Type "source ~/.bashrc"
Enter
Type "clear"
Enter
Show

# Enter Nix shell
Sleep 1s
Type "nix develop"
Enter
Sleep 2s

# Enter Pixi
Type "pixi shell"
Enter
Sleep 2s

# Build
Type "make build"
Enter
Sleep 7s

# Install
Type "uv pip install dist/wepy-1.1.0-py2.py3-none-any.whl"
Enter
Sleep 2s

# Run
Type "python info/examples/PySCF/source/revo_pyscf_alanine.py"
Enter
Sleep 7s
