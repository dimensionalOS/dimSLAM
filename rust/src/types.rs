// Copyright 2026 Dimensional Inc.
// SPDX-License-Identifier: Apache-2.0

//! The data the odometry takes in and gives back.
//!
//! Stamps are nanoseconds on whatever clock the caller feeds in; only differences matter.

use nalgebra::{Isometry3, Matrix6, Vector3};

/// A camera or depth image.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct ImageFrame {
    pub timestamp_ns: i64,
    pub frame_id: String,
    pub width: i32,
    pub height: i32,
    /// ROS names. Anything but `mono8` is fed to cuVSLAM as three-channel colour.
    pub encoding: String,
    /// Bytes per row, which may exceed `width` times the pixel size.
    pub step: i32,
    pub data: Vec<u8>,
}

/// Pinhole intrinsics plus distortion.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct CameraModel {
    pub timestamp_ns: i64,
    pub frame_id: String,
    pub width: i32,
    pub height: i32,
    /// `plumb_bob` ordering is assumed: k1, k2, p1, p2, k3.
    pub distortion: Vec<f64>,
    /// Row-major 3x3 intrinsics: fx, 0, cx, 0, fy, cy, 0, 0, 1.
    pub intrinsics: [f64; 9],
}

#[derive(Debug, Clone, Default, PartialEq)]
pub struct ImuSample {
    pub timestamp_ns: i64,
    pub frame_id: String,
    pub angular_velocity: Vector3<f64>,
    pub linear_acceleration: Vector3<f64>,
}

/// cuVSLAM's inertial mode needs the sensor's noise model up front; it cannot learn it.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct ImuNoiseModel {
    pub frame_id: String,
    pub gyro_noise_density: f64,
    pub gyro_random_walk: f64,
    pub accel_noise_density: f64,
    pub accel_random_walk: f64,
    pub frequency: f64,
}

#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct Twist {
    pub linear: Vector3<f64>,
    pub angular: Vector3<f64>,
}

/// A pose of `child_frame_id` in `frame_id`, with the twist expressed in `child_frame_id`.
#[derive(Debug, Clone, PartialEq)]
pub struct OdometryEstimate {
    pub timestamp_ns: i64,
    pub frame_id: String,
    pub child_frame_id: String,
    pub pose: Isometry3<f64>,
    /// xyz then rpy.
    pub pose_covariance: Matrix6<f64>,
    pub twist: Twist,
    pub twist_covariance: Matrix6<f64>,
}

impl Default for OdometryEstimate {
    fn default() -> Self {
        Self {
            timestamp_ns: 0,
            frame_id: String::new(),
            child_frame_id: String::new(),
            pose: Isometry3::identity(),
            pose_covariance: Matrix6::zeros(),
            twist: Twist::default(),
            twist_covariance: Matrix6::zeros(),
        }
    }
}

/// Range-gated depth points in the depth sensor's own frame.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct PointCloud {
    pub timestamp_ns: i64,
    pub frame_id: String,
    pub points: Vec<[f32; 3]>,
}

/// Both cores look up rigid mounts only, so a single latest-value query is enough; they
/// re-query until it answers and cache the result.
pub trait TfLookup {
    /// The transform taking a point in `child` to one in `parent`, or `None` if unconnected.
    fn latest(&self, parent: &str, child: &str) -> Option<Isometry3<f64>>;
}

impl<F> TfLookup for F
where
    F: Fn(&str, &str) -> Option<Isometry3<f64>>,
{
    fn latest(&self, parent: &str, child: &str) -> Option<Isometry3<f64>> {
        self(parent, child)
    }
}
