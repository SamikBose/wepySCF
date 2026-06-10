# Wepy Development Environments

Below are steps for setting up development environments for Wepy locally. Instructions only for CPU currently.

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
