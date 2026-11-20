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
pub use scan_sync::{header_seconds, Level, LevelConfig, ScanSync, SyncedScan};

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
///
/// Noise comes either as per-reading variances (`gyro_variance`, `accel_variance`, and bias
/// random walks per s^2, Point-LIO's convention) or, as dimSLAM main's configs give it, as
/// continuous-time densities. A density above zero replaces its per-reading value once the IMU's
/// rate `f` is measured over its first `init_samples` samples: variance = density^2 * f and bias
/// random walk = random_walk^2 * f (Point-LIO scales its bias noise by dt^2, so the continuous
/// growth random_walk^2 * dt becomes random_walk^2 / dt per s^2). Until then a further IMU's
/// samples are not used; the first IMU's only level the filter.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct ImuConfig {
    pub frame_id: String,
    /// rad^2/s^2: variance of one gyro reading.
    pub gyro_variance: f64,
    /// m^2/s^4: variance of one accelerometer reading.
    pub accel_variance: f64,
    /// rad/s and g: readings within `clip_fraction` of these are clipped and say nothing; 0
    /// trusts every reading.
    pub gyro_clip: f64,
    pub accel_clip_g: f64,
    /// Fraction of a clip level from which a reading counts as clipped (Point-LIO's 0.99).
    pub clip_fraction: f64,
    /// Random-walk variances of the gyro and accelerometer biases, per s^2.
    pub gyro_bias_random_walk: f64,
    pub accel_bias_random_walk: f64,
    /// Continuous-time noise densities (rad/s/sqrt(Hz), m/s^2/sqrt(Hz)) and bias random walks
    /// (rad/s^2/sqrt(Hz), m/s^3/sqrt(Hz)), as dimSLAM main reads them; 0 keeps the per-reading
    /// values above.
    pub gyro_noise_density: f64,
    pub gyro_random_walk: f64,
    pub accel_noise_density: f64,
    pub accel_random_walk: f64,
    /// Samples averaged to level the filter (the first IMU) and to measure the rate a density
    /// is converted at. Point-LIO's 100.
    pub init_samples: u32,
    /// rad/s: a sample turning faster than this restarts levelling, so the filter levels on a
    /// body at rest; 0 never restarts (Point-LIO's behaviour).
    pub init_gyro_limit: f64,
    /// s added to this IMU's stamps to put them on the clock the lidars' points are on (e.g. an
    /// IMU stamped by its host after a serial read and a DDS hop, against PTP-stamped lidars).
    pub time_offset_s: f64,
}

impl Default for ImuConfig {
    fn default() -> Self {
        Self {
            frame_id: String::new(),
            gyro_variance: 0.01,
            accel_variance: 0.01,
            gyro_clip: 35.0,
            accel_clip_g: 3.0,
            clip_fraction: 0.99,
            gyro_bias_random_walk: 0.0001,
            accel_bias_random_walk: 0.0001,
            gyro_noise_density: 0.0,
            gyro_random_walk: 0.0,
            accel_noise_density: 0.0,
            accel_random_walk: 0.0,
            init_samples: 100,
            init_gyro_limit: 0.0,
            time_offset_s: 0.0,
        }
    }
}

/// One lidar: the frame its scans carry, placed relative to the output frame by tf, and how its
/// points are thinned, matched and waited for. The defaults are Point-LIO's.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct LidarConfig {
    pub frame_id: String,
    /// m^2: variance of one point's distance to its plane.
    pub point_variance: f64,
    /// A scan of this lidar is waited for until it is this many of its own scan periods past
    /// its newest one, measured by the other lidars' scans; later, a scan of another lidar goes
    /// on without it.
    pub max_wait_periods: f64,
    /// m: nearer points are the robot itself.
    pub min_range: f64,
    /// Keep every this many usable points of a scan.
    pub keep_every: u32,
    /// m: a scan is thinned to one point per cube of this side; 0 keeps every point.
    pub scan_cube: f32,
    /// m: a point repeating the scan's previous one within this on every axis is dropped.
    pub duplicate_tolerance: f64,
    /// Map neighbors a plane is fitted through (at least 3).
    pub plane_points: usize,
    /// m: a fitted plane is used only when all its points lie this close to it.
    pub plane_tolerance: f32,
    /// A point is used only when its range exceeds this times the square of its distance to
    /// its plane: near points must lie closer to it than far ones.
    pub range_per_distance_sq: f64,
    /// m: map points farther than this are not neighbors.
    pub neighbor_range: f64,
}

impl Default for LidarConfig {
    fn default() -> Self {
        Self {
            frame_id: String::new(),
            point_variance: 0.01,
            max_wait_periods: 2.0,
            min_range: 0.5,
            keep_every: 3,
            scan_cube: 0.2,
            duplicate_tolerance: 1e-7,
            plane_points: 5,
            plane_tolerance: 0.1,
            range_per_distance_sq: 81.0,
            neighbor_range: 5.0,
        }
    }
}

/// 1-sigma uncertainty the filter starts with, per state block (dimSLAM main's names, plus the
/// blocks Point-LIO's state adds). The defaults are Point-LIO's: a variance of 0.01 everywhere
/// but gravity (1e-4) and the biases (1e-3), which `bias` applies to every IMU's.
#[derive(Clone, Copy, Debug, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct InitialStds {
    /// m
    pub position: f64,
    /// m/s
    pub velocity: f64,
    /// rad
    pub rotation: f64,
    /// rad/s and m/s^2 (gyro and accelerometer alike)
    pub bias: f64,
    /// rad and m: the lidar mount states
    pub mount_rotation: f64,
    pub mount_translation: f64,
    /// rad/s
    pub angular_velocity: f64,
    /// m/s^2
    pub specific_force: f64,
    /// m/s^2
    pub gravity: f64,
}

impl Default for InitialStds {
    fn default() -> Self {
        let std = 0.01f64.sqrt();
        Self {
            position: std,
            velocity: std,
            rotation: std,
            bias: 0.001f64.sqrt(),
            mount_rotation: std,
            mount_translation: std,
            angular_velocity: std,
            specific_force: std,
            gravity: 0.0001f64.sqrt(),
        }
    }
}

/// The map scans are matched against. The defaults are Point-LIO's.
#[derive(Clone, Copy, Debug, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct MapConfig {
    /// m: a scan point is added only where the map has none in its cube of this side.
    pub spacing: f64,
    /// m: side of the map's hash cubes; neighbors are searched in a point's cube and the six
    /// sharing its faces.
    pub cube: f32,
    /// Cubes kept; past this the least recently touched goes.
    pub capacity: usize,
    /// Points the first scans must give before the map is used.
    pub seed_points: usize,
    /// A scan grows the map only when it kept more than this many points.
    pub grow_min_points: usize,
}

impl Default for MapConfig {
    fn default() -> Self {
        Self {
            spacing: 0.5,
            cube: 2.0,
            capacity: 1_000_000,
            seed_points: 10,
            grow_min_points: 4,
        }
    }
}

/// Each scan's points corrected by the filter's own pose at each point's instant, to one
/// instant (the scan's end), as a cloud per scan. Off by default.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct DeskewConfig {
    pub enabled: bool,
    /// "output": the body at the scan's end; "odom": the filter's world.
    pub frame: DeskewFrame,
    /// m: one point per cube of this side; 0 keeps every point.
    pub voxel: f32,
}

impl Default for DeskewConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            frame: DeskewFrame::Output,
            voxel: 0.0,
        }
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum DeskewFrame {
    #[default]
    Output,
    Odom,
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
