# dimSLAM

SLAM stack for dimos. Rust only: the C++ cuVSLAM fork, its extern-"C" shim, and
the per-platform SDK packaging live in
[cu_vslam_rs](https://github.com/jeff-hykin/cu_vslam_rs).

- `rust/` — the `dim_slam` library crate: one odometry filter (Point-LIO's
  estimator, grown to take any number of IMUs, lidars, cameras through cuVSLAM
  (via the `cu_vslam_rs` crate) and other odometry sources, each placed by tf).
  It is plain synchronous Rust over plain Rust types, owning no transport or
  threads, so anything can drive it; dimos supplies its own LCM wrapper. Needs
  `CUVSLAM_SDK_DIR` (a dir with `include/cuvslam/cuvslam2.h` and
  `lib/libcuvslam.*`) to build.
- `flake.nix` — re-exports the cuVSLAM SDKs this arch can run as `sdk-<variant>`
  (`x86_64-cuda12` and `x86_64-cuda13` on x86 linux, `orin` and `thor` on arm
  linux, `metal` on arm macOS), and a devShell with `CUVSLAM_SDK_DIR` already
  set to this machine's default.

```rust
use dim_slam::{DimSlamCore, DimSlamCoreConfig, ImuConfig, LidarConfig, SourceConfig, SourceKind};

// The output frame is the body; every IMU, lidar and source frame is placed relative to it
// by tf. Sources are `pose` (a non-drifting world frame), `twist` (body velocity) or
// `pose_delta` (a drifting pose, fused by its motion only).
let mut slam = DimSlamCore::new(DimSlamCoreConfig {
    output_frame_id: "base_link".into(),
    imus: vec![ImuConfig { frame_id: "imu_link".into(), ..Default::default() }],
    lidars: vec![LidarConfig { frame_id: "lidar_link".into() }],
    odom_sources: vec![SourceConfig {
        parent_frame_id: "wheel_odom".into(),
        child_frame_id: "base_link".into(),
        kind: SourceKind::Twist,
        twist_variances: [1e-4, 1e-4, 1e-4, 1e-4, 1e-4, 1e-4],
        ..Default::default()
    }],
    ..Default::default()
})?;
slam.handle_imu(&imu_sample, &tf);
slam.handle_cloud(&scan, &tf);
slam.handle_source(&wheel_odometry, &tf);
if let Some(estimate) = slam.maybe_publish() { /* estimate.pose */ }
```

A binary that links this crate has to embed the SDK's rpath itself; cargo does
not pass `rustc-link-arg` down from a dependency's build script.
