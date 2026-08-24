{
  description = "dimSLAM: the Rust dim_slam module, built on the cu_vslam_rs crate and its per-platform cuVSLAM SDKs";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
    flake-utils.url = "github:numtide/flake-utils";
    cu-vslam-rs.url = "github:jeff-hykin/cu_vslam_rs";
    cu-vslam-rs.inputs.nixpkgs.follows = "nixpkgs";
    cu-vslam-rs.inputs.flake-utils.follows = "flake-utils";
  };

  outputs = { self, nixpkgs, flake-utils, cu-vslam-rs, ... }:
    flake-utils.lib.eachDefaultSystem (system:
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

        # The crates' git dependencies, pre-fetched so the sandboxed cargo build
        # needs no network. Both dimos-module crates come from the same dimos
        # checkout, so they share a hash. Bump alongside Cargo.lock.
        rustGitDepHashes = {
          "dimos-module-0.1.0" = "sha256-nMnCk9RbDKhOUZRCQUCjZVVPeQY8HrwFLItaZ64Q1KY=";
          "dimos-module-macros-0.1.0" = "sha256-nMnCk9RbDKhOUZRCQUCjZVVPeQY8HrwFLItaZ64Q1KY=";
          "lcm-msgs-0.1.0" = "sha256-GGkx4Mn6NYP6KZecmoRLKGWIih/+y8OgNn12DeXX6n8=";
        };

        moduleFor = variant: let sdkPackage = sdkPackages."sdk-${variant}"; in
          pkgs.rustPlatform.buildRustPackage {
            pname = "dim_slam";
            version = "0.1.0";
            src = ./rust;
            cargoLock = {
              lockFile = ./rust/Cargo.lock;
              outputHashes = rustGitDepHashes;
            };
            # cu_vslam_rs's build.rs compiles its shim against this SDK.
            env.CUVSLAM_SDK_DIR = sdkPackage;
            # The crate's build.rs cannot set the rpath: cargo drops rustc-link-arg
            # from dependency build scripts, so the final binary must embed it here.
            env.RUSTFLAGS = "-C link-arg=-Wl,-rpath,${sdkPackage}/lib";
            nativeBuildInputs = pkgs.lib.optionals isDarwin [ pkgs.makeWrapper ];
            # The test binary links libcuvslam, whose CUDA runtime wants a GPU
            # driver the build sandbox lacks; unit tests run via plain cargo test.
            doCheck = false;
            # The SDK ships every kernel already compiled, but its store path is
            # read-only so CuMetal cannot use it as the normal cache; this is the
            # read-only lookup it consults first.
            postInstall = pkgs.lib.optionalString isDarwin ''
              wrapProgram $out/bin/dim_slam \
                --set-default CUMETAL_PREBUILT_CACHE_DIR ${sdkPackage}/share/cumetal-cache
            '';
          };
      in {
        # One dim_slam package per SDK variant on this arch, plus the SDKs
        # themselves re-exported under the same sdk-<variant> names.
        packages = nixpkgs.lib.genAttrs variants moduleFor
          // sdkPackages
          // { default = moduleFor defaultVariant; };

        devShells.default = let sdkPackage = sdkPackages."sdk-${defaultVariant}"; in
          pkgs.mkShell {
            packages = [ pkgs.cargo pkgs.rustc pkgs.clippy pkgs.rustfmt ];
            CUVSLAM_SDK_DIR = sdkPackage;
            # Same reason as moduleFor: without the rpath the test binary cannot find
            # libcuvslam and aborts before the first test runs.
            RUSTFLAGS = "-C link-arg=-Wl,-rpath,${sdkPackage}/lib";
          };
      });
}
