{
  description = "A very basic flake";

  inputs = {
    nixpkgs.url = "github:nixos/nixpkgs?ref=nixos-unstable";
    flake-utils.url = "github:numtide/flake-utils";
  };

  outputs = inputs @ {self, ...}:
    with inputs;
      flake-utils.lib.eachDefaultSystem (system: let
        pkgs = import nixpkgs {
          inherit system;
          overlay = [];
        };
      in {
        devShell = pkgs.mkShell {
          buildInputs = [
            pkgs.python312Packages.opencv-python-headless
          ];
          packages = with pkgs; [
            maturin
            uv
            cargo-bloat
            cargo-edit
            cargo-outdated
            cargo-udeps
            cargo-watch
            rust-analyzer
            ffmpeg
            # libGL
            # glib
          ];
          shellHook = ''
            export LD_LIBRARY_PATH=${pkgs.libGL}/lib:$LD_LIBRARY_PATH
            export LD_LIBRARY_PATH=${pkgs.glib}/lib:$LD_LIBRARY_PATH
            source .venv/bin/activate
          '';
        };
      });
}
