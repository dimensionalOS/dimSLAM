# dimSLAM

SLAM stack for dimos. Rust only: the C++ cuVSLAM fork, its extern-"C" shim, and
the per-platform SDK packaging live in
[cu_vslam_rs](https://github.com/jeff-hykin/cu_vslam_rs).

- `rust/` — the `dim_slam` library crate: cuVSLAM visual odometry (via the
  `cu_vslam_rs` crate) and an error-state Kalman fusion filter. Both cores are
  plain synchronous Rust over plain Rust types, owning no transport or threads,
  so anything can drive them; dimos supplies its own LCM wrapper. Needs
  `CUVSLAM_SDK_DIR` (a dir with `include/cuvslam/cuvslam2.h` and
  `lib/libcuvslam.*`) to build.
- `flake.nix` — re-exports the cuVSLAM SDKs this arch can run as `sdk-<variant>`
  (`x86_64-cuda12` and `x86_64-cuda13` on x86 linux, `orin` and `thor` on arm
  linux, `metal` on arm macOS), and a devShell with `CUVSLAM_SDK_DIR` already
  set to this machine's default.

```rust
use dim_slam::{FusionCore, OdometryFusionConfig};

let mut fusion = FusionCore::new(OdometryFusionConfig {
    source_frames: vec!["visual_odom".into()],
    source_pose_variances: vec![1e-4; 6],
    source_twist_variances: vec![0.0; 6],
    ..Default::default()
});
fusion.handle_source(&visual_odometry);
if let Some(estimate) = fusion.maybe_publish() { /* estimate.pose */ }
```

A binary that links this crate has to embed the SDK's rpath itself; cargo does
not pass `rustc-link-arg` down from a dependency's build script.
