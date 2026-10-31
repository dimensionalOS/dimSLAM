// Copyright 2026 Dimensional Inc.
// SPDX-License-Identifier: Apache-2.0
//
// The odometry filter (Point-LIO's estimator, grown to take any set of sensors) and the
// configuration of what it is fed: IMUs, lidars and odometry sources, each placed by tf.

pub mod lio_filter;
pub mod lio_odometry;
pub mod plane_fit;
mod point_map;
mod scan_points;
mod scan_sync;

use serde::{Deserialize, Serialize};

pub use point_map::{to_world, PointMap};
pub use scan_points::{surface_points, voxel_centroids, SurfacePoint};
pub use scan_sync::{header_seconds, Level, ScanSync, SyncedScan, STANDARD_GRAVITY};

/// What a source's pose means to the filter. Its twist, wherever twist_variances are set,
/// is a body-frame velocity measurement whatever the kind.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SourceKind {
    /// The pose in a world frame of the source's own that does not drift (e.g. lidar-inertial
    /// odometry). Its first message aligns that frame to the filter's; later ones are poses.
    Pose,
    /// The pose is not used, only the twist (e.g. wheel odometry).
    Twist,
    /// A pose that drifts (e.g. a visual tracker): only the motion between consecutive messages
    /// is fused, as the body's velocity and angular velocity over that interval.
    #[default]
    PoseDelta,
}

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct SourceConfig {
    /// The transform this source's estimates carry. Both halves are needed: two sources can
    /// share a parent. The child may be any frame tf places relative to the output frame.
    pub parent_frame_id: String,
    pub child_frame_id: String,
    pub kind: SourceKind,
    /// Measurement noise, [x y z roll pitch yaw]: <0 message covariance, 0 drop, >0 fixed;
    /// the floor of the spread the filter learns from the source's residuals. For `pose` the
    /// noise of the pose itself, position along the world axes; for `pose_delta` the noise of
    /// one message-to-message motion, in the body frame.
    pub pose_variances: [f64; 6],
    /// [vx vy vz wx wy wz], same convention, body frame.
    pub twist_variances: [f64; 6],
    /// Zero-velocity update: a twist at or under `still_speed` (m/s) and `still_angular_speed`
    /// (rad/s) is fused as zero velocity with `still_variance` in every axis, in place of the
    /// twist itself. 0 disables.
    pub still_variance: f64,
    pub still_speed: f64,
    pub still_angular_speed: f64,
}

/// One IMU: the frame its samples carry (placed relative to the output frame by tf) and how it
/// is read. The defaults are Point-LIO's.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct ImuConfig {
    pub frame_id: String,
    /// rad^2/s^2: variance of one gyro reading.
    pub gyro_variance: f64,
    /// m^2/s^4: variance of one accelerometer reading.
    pub accel_variance: f64,
    /// rad/s and g: readings within 1% of these are clipped and say nothing; 0 trusts every
    /// reading.
    pub gyro_clip: f64,
    pub accel_clip_g: f64,
    /// Random-walk variances of the gyro and accelerometer biases, per s^2.
    pub gyro_bias_random_walk: f64,
    pub accel_bias_random_walk: f64,
}

impl Default for ImuConfig {
    fn default() -> Self {
        Self {
            frame_id: String::new(),
            gyro_variance: 0.01,
            accel_variance: 0.01,
            gyro_clip: 35.0,
            accel_clip_g: 3.0,
            gyro_bias_random_walk: 0.0001,
            accel_bias_random_walk: 0.0001,
        }
    }
}

/// One lidar: the frame its scans carry, placed relative to the output frame by tf.
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct LidarConfig {
    pub frame_id: String,
}

/// How fast the body's motion can change unannounced: random-walk variances, per s^2, of the
/// velocity, angular velocity and specific force. The defaults are Point-LIO's, which an IMU
/// at hundreds of hertz keeps in check; a body with no IMU wants its own.
#[derive(Clone, Copy, Debug, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct ProcessNoise {
    pub velocity: f64,
    pub angular_velocity: f64,
    pub specific_force: f64,
}

impl Default for ProcessNoise {
    fn default() -> Self {
        Self {
            velocity: 20.0,
            angular_velocity: 1000.0,
            specific_force: 500.0,
        }
    }
}
