{
  description = "dimSLAM: the Rust dim_slam library, built on the cu_vslam_rs crate and its per-platform cuVSLAM SDKs";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
    flake-utils.url = "github:numtide/flake-utils";
    cu-vslam-rs.url = "github:jeff-hykin/cu_vslam_rs";
    cu-vslam-rs.inputs.nixpkgs.follows = "nixpkgs";
    cu-vslam-rs.inputs.flake-utils.follows = "flake-utils";
    crate2nix.url = "github:nix-community/crate2nix";
    crate2nix.inputs.nixpkgs.follows = "nixpkgs";
  };

  outputs = { self, nixpkgs, flake-utils, cu-vslam-rs, crate2nix, ... }:
    # Not eachDefaultSystem: nixpkgs 26.11 dropped x86_64-darwin, and merely naming
    # it is an eval error.
    flake-utils.lib.eachSystem [ "aarch64-darwin" "aarch64-linux" "x86_64-linux" ] (system:
      let
        isDarwin = nixpkgs.lib.hasSuffix "-darwin" system;
        pkgs = import nixpkgs {
          inherit system;
          config = { allowUnfree = true; cudaSupport = !isDarwin; };
        };

        # cuVSLAM SDKs come from the cu_vslam_rs flake: one sdk-<variant> package
        # per build available on this system (see that flake for the variant list).
        sdkPackages = nixpkgs.lib.filterAttrs (name: _: nixpkgs.lib.hasPrefix "sdk-" name)
          cu-vslam-rs.packages.${system};
        variants = map (nixpkgs.lib.removePrefix "sdk-") (builtins.attrNames sdkPackages);

        # What a machine that says nothing about itself gets. CUDA 12 on both linux
        # arches: it is what the drivers in the field are, and a 13 driver runs a 12
        # build.
        defaultVariant = {
          aarch64-darwin = "metal";
          aarch64-linux = "orin";
        }.${system} or "x86_64-cuda12";

        # One derivation per crate rather than one vendored blob, so a dependency
        # bump only rebuilds what changed and the SDK variants share the rest.
        generatedCargoNix = crate2nix.tools.${system}.generatedCargoNix {
          name = "dim_slam";
          src = ./rust;
        };

        checkFor = variant: let sdkPackage = sdkPackages."sdk-${variant}"; in
          (import generatedCargoNix {
            inherit pkgs;
            buildRustCrateForPkgs = cratePkgs: cratePkgs.buildRustCrate.override {
              defaultCrateOverrides = cratePkgs.defaultCrateOverrides // {
                # cu_vslam_rs's build.rs compiles its shim against this SDK.
                cu_vslam_rs = _: { CUVSLAM_SDK_DIR = sdkPackage; };
              };
            };
          # The rlib, not the empty default output a lib crate leaves behind.
          }).rootCrate.build.lib;
      in {
        # The library is consumed from cargo; what nix is for here is the SDK it builds against.
        packages = sdkPackages // { default = sdkPackages."sdk-${defaultVariant}"; };

        # One compile per SDK variant available on this arch, so a broken variant
        # shows up here rather than in whatever downstream binary links it.
        checks = nixpkgs.lib.genAttrs variants checkFor;

        devShells.default = let sdkPackage = sdkPackages."sdk-${defaultVariant}"; in
          pkgs.mkShell ({
            packages = [ pkgs.cargo pkgs.rustc pkgs.clippy pkgs.rustfmt ];
            CUVSLAM_SDK_DIR = sdkPackage;
          } // pkgs.lib.optionalAttrs isDarwin {
            # The SDK ships every kernel already compiled, but its store path is
            # read-only so CuMetal cannot use it as the normal cache; this is the
            # read-only lookup it consults first.
            CUMETAL_PREBUILT_CACHE_DIR = "${sdkPackage}/share/cumetal-cache";
          });
      });
}
