// Copyright 2026 Dimensional Inc.
// SPDX-License-Identifier: Apache-2.0

//! Odometry from any set of sensors: IMUs, lidars, cameras and other odometry sources, all
//! fused in one filter (Point-LIO's estimator, grown to take them all).
//!
//! [`DimSlamCore`] is the whole thing: feed it frames, IMU samples, scans and any other
//! odometry source, then ask for the fused pose. Every sensor is placed relative to the output
//! frame (the body) by tf. [`CuvslamCore`], the camera tracker, is also public on its own.
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
//!     ..Default::default()
//! })
//! .expect("a usable config");
//! // Mounts come from tf; the source reports the output frame itself here.
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
//! let estimate = slam.maybe_publish().expect("the wheels moved the filter");
//! assert!((estimate.pose.translation.x - 0.5).abs() < 0.05);
//! ```

pub mod cuvslam_odometry;
pub mod dim_slam_core;
#[doc(hidden)]
pub mod log;
pub mod odometry_fusion;
pub mod types;

pub use cuvslam_odometry::{CameraConfig, CuvslamCore, CuvslamOdometryConfig};
pub use dim_slam_core::{DimSlamCore, DimSlamCoreConfig};
pub use odometry_fusion::{ImuConfig, LidarConfig, ProcessNoise, SourceConfig, SourceKind};
pub use types::{
    CameraModel, ImageFrame, ImuNoiseModel, ImuSample, LidarScan, OdometryEstimate, PointCloud, TfLookup,
    Twist,
};

/// Poses cross the API as `nalgebra` types, so a caller need not pin the version itself.
pub use nalgebra;
