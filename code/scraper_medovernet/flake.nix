{
  description = "Python development environment";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-23.11";
    flake-utils.url = "github:numtide/flake-utils";
  };

  outputs = { self, nixpkgs, flake-utils, ... }@inputs:
    flake-utils.lib.eachDefaultSystem (system:
      let
        pkgs = import nixpkgs {
          inherit system;

        config = {
            allowUnfree = true;
          };
        };
        pythonEnv = pkgs.python3.withPackages (ps: with ps; [
          matplotlib
          numpy
          seaborn
          pyyaml
          selenium
          requests
          tqdm
        ]);
      in {
        devShells.default = pkgs.mkShell {
          buildInputs = [ pythonEnv pkgs.chromedriver pkgs.google-chrome]; 
        };
      });
}
