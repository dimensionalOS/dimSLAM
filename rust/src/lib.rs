// Copyright 2026 Dimensional Inc.
// SPDX-License-Identifier: Apache-2.0

//! Visual-inertial odometry: a cuVSLAM front end and an error-state Kalman fuser.
//!
//! [`CuvslamCore`] turns camera, depth and IMU frames into a pose; [`FusionCore`] blends that pose
//! with any other odometry source against IMU propagation. Either runs on its own.
//!
//! Both are synchronous and own no threads or transport: feed them frames, ask for output.
//!
//! ```
//! use dim_slam::nalgebra::Isometry3;
//! use dim_slam::{FusionCore, OdometryEstimate, OdometryFusionConfig, SourceConfig};
//!
//! // An odometry source is identified by the transform its estimates carry:
//! // the message's frame_id -> child_frame_id.
//! let mut fusion = FusionCore::new(OdometryFusionConfig {
//!     odom_sources: vec![SourceConfig {
//!         parent_frame_id: "visual_odom".into(),
//!         child_frame_id: "base_link".into(),
//!         pose_variances: [1e-4; 6],
//!         twist_variances: [0.0; 6],
//!     }],
//!     ..Default::default()
//! });
//! for (step, timestamp_ns) in [0, 20_000_000, 40_000_000].into_iter().enumerate() {
//!     fusion.handle_source(&OdometryEstimate {
//!         timestamp_ns,
//!         frame_id: "visual_odom".into(),
//!         child_frame_id: "base_link".into(),
//!         pose: Isometry3::translation(step as f64 * 0.1, 0.0, 0.0),
//!         ..Default::default()
//!     });
//! }
//! let estimate = fusion.maybe_publish().expect("a publish period has elapsed");
//! assert!((estimate.pose.translation.x - 0.2).abs() < 1e-9);
//! ```

pub mod cuvslam_odometry;
#[doc(hidden)]
pub mod log;
pub mod odometry_fusion;
pub mod types;

pub use cuvslam_odometry::{CameraConfig, CuvslamCore, CuvslamOdometryConfig};
pub use odometry_fusion::{FusionCore, ImuConfig, OdometryFusionConfig, SourceConfig};
pub use types::{
    CameraModel, ImageFrame, ImuNoiseModel, ImuSample, OdometryEstimate, PointCloud, TfLookup,
    Twist,
};

/// Poses cross the API as `nalgebra` types, so a caller need not pin the version itself.
pub use nalgebra;
