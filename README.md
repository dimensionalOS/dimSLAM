# dimSLAM

SLAM stack for dimos.

- `cuvslam/` — our cuVSLAM lineage (NVIDIA cuVSLAM v17 + odometry state
  serialization). Licensed under the NVIDIA Community License (`cuvslam/LICENSE`).
- `rust/cuvslam_odometry/` — the dimos odometry module built on top of it: Rust,
  with a small extern-"C" shim (`shim/`) as the only C++. Needs `CUVSLAM_SDK_DIR`
  (a dir with `include/cuvslam/cuvslam2.h` and `lib/libcuvslam.*`) to build.
- `rust/odometry_fusion/` — the fusion filter downstream of the visual odometry
  (IMU propagation corrected by any number of odometry sources). Pure Rust,
  no cuVSLAM dependency.
- `flake.nix` — builds everything: `nix build .#<variant>` (variants:
  `x86_64-cuda12`, `x86_64-cuda13`, `orin`, `thor`, `metal`; fork-source builds
  for x86_64-cuda12 and orin, NVIDIA release tarballs for the rest). Each variant
  packages both module binaries.

dimos consumes this repo by git tag:
`nix build github:dimensionalOS/dimSLAM/<tag>#<variant>`
