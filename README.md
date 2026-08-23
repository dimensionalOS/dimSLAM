# dimSLAM

SLAM stack for dimos. Rust only: the C++ cuVSLAM fork, its extern-"C" shim, and
the per-platform SDK packaging live in
[cu_vslam_rs](https://github.com/jeff-hykin/cu_vslam_rs).

- `rust/` — the `dim_slam` dimos module: cuVSLAM visual odometry (via the
  `cu_vslam_rs` crate) feeding an error-state Kalman fusion filter in-process.
  Needs `CUVSLAM_SDK_DIR` (a dir with `include/cuvslam/cuvslam2.h` and
  `lib/libcuvslam.*`) to build.
- `flake.nix` — builds `dim_slam` per SDK variant: `nix build .#<variant>`
  (variants: `x86_64-cuda12`, `x86_64-cuda13`, `orin`, `thor`, `metal`). The
  SDKs come from the cu_vslam_rs flake input.

dimos consumes this repo by git tag:
`nix build github:dimensionalOS/dimSLAM/<tag>#<variant>`
