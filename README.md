# dimSLAM

SLAM stack for dimos.

- `cuvslam/` — our cuVSLAM lineage (NVIDIA cuVSLAM v17 + odometry state
  serialization). Licensed under the NVIDIA Community License (`cuvslam/LICENSE`).
- `rust/` — the `dim_slam` dimos module: cuVSLAM visual odometry (via a small
  extern-"C" shim in `shim/`, the only C++) feeding an error-state Kalman fusion
  filter in-process. Needs `CUVSLAM_SDK_DIR` (a dir with
  `include/cuvslam/cuvslam2.h` and `lib/libcuvslam.*`) to build.
- `flake.nix` — builds everything: `nix build .#<variant>` (variants:
  `x86_64-cuda12`, `x86_64-cuda13`, `orin`, `thor`, `metal`; fork-source builds
  for x86_64-cuda12 and orin, NVIDIA release tarballs for the rest).

dimos consumes this repo by git tag:
`nix build github:dimensionalOS/dimSLAM/<tag>#<variant>`
