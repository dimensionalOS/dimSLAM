// Copyright 2026 Dimensional Inc.
// SPDX-License-Identifier: Apache-2.0

//! Visual-inertial odometry: a cuVSLAM front end and an error-state Kalman fuser.
//!
//! [`DimSlamCore`] is the whole thing: feed it frames, IMU samples and any other odometry
//! source, then ask for the fused pose. It owns the tracker's wiring into the filter.
//!
//! The two halves are also public on their own. [`CuvslamCore`] turns camera, depth and IMU
//! frames into a pose; [`FusionCore`] blends a pose with any other odometry source against IMU
//! propagation. Reach for them only when the wiring itself has to differ.
//!
//! All are synchronous and own no threads or transport.
//!
//! ```
//! use dim_slam::nalgebra::Isometry3;
//! use dim_slam::{DimSlamCore, DimSlamCoreConfig, OdometryEstimate, SourceConfig};
//!
//! // An odometry source is identified by the transform its estimates carry:
//! // the message's frame_id -> child_frame_id.
//! let mut slam = DimSlamCore::new(DimSlamCoreConfig {
//!     visual_odom_pose_variances: [1e-4; 6],
//!     odom_sources: vec![SourceConfig {
//!         parent_frame_id: "wheel_odom".into(),
//!         child_frame_id: "base_link".into(),
//!         pose_variances: [1e-4; 6],
//!         twist_variances: [0.0; 6],
//!     }],
//!     ..Default::default()
//! })
//! .expect("a usable config");
//! for (step, timestamp_ns) in [0, 20_000_000, 40_000_000].into_iter().enumerate() {
//!     slam.handle_source(&OdometryEstimate {
//!         timestamp_ns,
//!         frame_id: "wheel_odom".into(),
//!         child_frame_id: "base_link".into(),
//!         pose: Isometry3::translation(step as f64 * 0.1, 0.0, 0.0),
//!         ..Default::default()
//!     });
//! }
//! let estimate = slam.maybe_publish().expect("a publish period has elapsed");
//! assert!((estimate.pose.translation.x - 0.2).abs() < 1e-9);
//! ```

pub mod cuvslam_odometry;
pub mod dim_slam_core;
#[doc(hidden)]
pub mod log;
pub mod odometry_fusion;
pub mod types;

pub use cuvslam_odometry::{CameraConfig, CuvslamCore, CuvslamOdometryConfig};
pub use dim_slam_core::{DimSlamCore, DimSlamCoreConfig};
pub use odometry_fusion::{FusionCore, ImuConfig, InitialStds, OdometryFusionConfig, SourceConfig};
pub use types::{
    CameraModel, ImageFrame, ImuNoiseModel, ImuSample, OdometryEstimate, PointCloud, TfLookup,
    Twist,
};

/// Poses cross the API as `nalgebra` types, so a caller need not pin the version itself.
pub use nalgebra;
