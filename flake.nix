# https://wiki.nixos.org/wiki/Python

{
  description = "Pixi development environment";
  inputs = {
    flake-utils.url = "github:numtide/flake-utils";
  };
  outputs =
    { flake-utils, nixpkgs, ... }:
    flake-utils.lib.eachDefaultSystem (
      system:
      let
        pkgs = import nixpkgs { inherit system; };
        fhs = pkgs.buildFHSEnv {
          name = "pixi-env";

          targetPkgs = _: [
            pkgs.pixi
            pkgs.gnumake # make build
            pkgs.uv # For installing from source (uv pip install dist/wepy-*.whl)
          ];
        };
      in
      {
        devShell = fhs.env;
      }
    );
}
