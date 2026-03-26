{
    description = "ML Python environment for landmine detection";

    inputs = {
        nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
        flake-utils.url = "github:numtide/flake-utils";
    };

    outputs = { self, nixpkgs, flake-utils }:
        flake-utils.lib.eachDefaultSystem (system:
            let
                pkgs = import nixpkgs {
                    inherit system;
                    config = {
                        allowUnfree = true;
                        cudaSupport = true;
                    };
                };

                libPath = with pkgs; lib.makeLibraryPath [
                    stdenv.cc.cc.lib
                    zlib
                    glib
                    libX11
                    libXext
                    libXrender
                    libICE
                    libSM
                    libGL
                ];
            in {
                devShells.default = pkgs.mkShell {
                    buildInputs = with pkgs; [
                        python312
                        python312Packages.pip
                        python312Packages.virtualenv
                        python312Packages.jupyter
                        python312Packages.notebook
                        gcc
                        gnumake
                        pkg-config
                    ];

                    shellHook = ''
                        if [ ! -d ".venv" ]; then
                            echo "Creating new virtual environment"
                            python -m venv .venv
                        fi
                        source .venv/bin/activate

                        export LD_LIBRARY_PATH="${libPath}:/run/opengl-driver/lib:/run/opengl-driver-32/lib:$LD_LIBRARY_PATH"

                        echo "Venv has been activated"
                        echo "To install requirement packages use: pip install -r requirements.txt"
                    '';
                };
            }
        );
}