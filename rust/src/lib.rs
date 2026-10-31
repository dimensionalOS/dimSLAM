// Copyright 2026 Dimensional Inc.
// SPDX-License-Identifier: Apache-2.0

//! Visual-inertial odometry: a cuVSLAM front end and an error-state Kalman fuser.
//!
//! [`DimSlamCore`] is the whole thing: feed it frames, IMU samples and any other odometry
//! source, then ask for the fused pose. It owns the tracker's wiring into the filter.
//!
//! The two halves are also public on their own. [`CuvslamCore`] turns camera, depth and IMU
//! frames into a pose; [`FusionCore`] is one error-state Kalman filter over any odometry
//! sources, predicting by IMU or, without one, by constant velocity. Reach for them only when
//! the wiring itself has to differ.
//!
//! All are synchronous and own no threads or transport.
//!
//! ```
//! use dim_slam::nalgebra::Isometry3;
//! use dim_slam::nalgebra::Vector3;
//! use dim_slam::{DimSlamCore, DimSlamCoreConfig, OdometryEstimate, SourceConfig, SourceKind, Twist};
//!
//! // An odometry source is identified by the transform its estimates carry:
//! // the message's frame_id -> child_frame_id. Wheels measure the body's velocity.
//! let mut slam = DimSlamCore::new(DimSlamCoreConfig {
//!     visual_odom_pose_variances: [1e-4; 6],
//!     odom_sources: vec![SourceConfig {
//!         parent_frame_id: "wheel_odom".into(),
//!         child_frame_id: "base_link".into(),
//!         kind: SourceKind::Twist,
//!         twist_variances: [1e-4, 1e-4, 0.0, 0.0, 0.0, 1e-4],
//!         ..Default::default()
//!     }],
//!     planar: true,
//!     ..Default::default()
//! })
//! .expect("a usable config");
//! // Mounts come from tf; the fusion engine does not need any here.
//! let tf = |_: &str, _: &str| -> Option<Isometry3<f64>> { None };
//! for step in 0..=50 {
//!     slam.handle_source(&OdometryEstimate {
//!         timestamp_ns: step * 20_000_000,
//!         frame_id: "wheel_odom".into(),
//!         child_frame_id: "base_link".into(),
//!         twist: Twist { linear: Vector3::new(0.5, 0.0, 0.0), angular: Vector3::zeros() },
//!         ..Default::default()
//!     }, &tf);
//! }
//! let estimate = slam.maybe_publish().expect("a publish period has elapsed");
//! assert!((estimate.pose.translation.x - 0.5).abs() < 0.05);
//! ```

pub mod cuvslam_odometry;
pub mod dim_slam_core;
#[doc(hidden)]
pub mod log;
pub mod odometry_fusion;
pub mod types;

pub use cuvslam_odometry::{CameraConfig, CuvslamCore, CuvslamOdometryConfig};
pub use dim_slam_core::{DimSlamCore, DimSlamCoreConfig, PointLioEngineConfig};
pub use odometry_fusion::{
    FusionCore, ImuConfig, InitialStds, OdometryFusionConfig, ProcessNoise, SourceConfig,
    SourceKind,
};
pub use types::{
    CameraModel, ImageFrame, ImuNoiseModel, ImuSample, LidarScan, OdometryEstimate, PointCloud, TfLookup,
    Twist,
};

/// Poses cross the API as `nalgebra` types, so a caller need not pin the version itself.
pub use nalgebra;
