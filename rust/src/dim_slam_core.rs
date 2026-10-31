// Copyright 2026 Dimensional Inc.
// SPDX-License-Identifier: Apache-2.0
//
// One filter fed by every sensor: any number of IMUs, lidars, cameras (through the cuVSLAM
// tracker) and odometry sources, each placed relative to the output frame by tf. The output
// frame is the body the filter estimates.

use serde::{Deserialize, Serialize};

use nalgebra::{Isometry3, Matrix3, Quaternion, Translation3, UnitQuaternion, Vector3};

use crate::cuvslam_odometry::{CameraConfig, CuvslamCore, CuvslamOdometryConfig};
use crate::odometry_fusion::lio_odometry::{point_lio_imu, LioConfig, LioOdometry, LioPose};
use crate::odometry_fusion::{
    header_seconds, ImuConfig, LidarConfig, ProcessNoise, SourceConfig, SourceKind,
    STANDARD_GRAVITY,
};
use crate::types::{
    CameraModel, ImageFrame, ImuSample, LidarScan, OdometryEstimate, PointCloud, TfLookup, Twist,
};
use crate::warn_throttled;

/// The tracker's own drifting world frame: it never leaves this struct.
const VISUAL_ODOM_FRAME_ID: &str = "visual_odom";

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct DimSlamCoreConfig {
    /// "stereo", "mono", "rgbd" or "multisensor". Mono is accurate only up to scale.
    pub camera_mode: String,
    /// In cuVSLAM's index order: the rig cameras first (two for stereo, the whole list for
    /// multisensor, one otherwise), then any settings-only streams such as an rgbd depth
    /// camera. Empty discovers the rig off camera_info; multisensor requires the list.
    pub cameras: Vec<CameraConfig>,
    /// Needs a libcuvslam built with ENFORCE_GPU=OFF; stock SDK binaries are GPU-only.
    pub use_gpu: bool,
    /// Empty means output_frame_id; NVIDIA's examples use the left camera's optical frame.
    pub rig_frame_id: String,
    /// Parent of the loop-closure correction, carried so a caller's frame naming survives.
    pub map_frame_id: String,
    /// Meters; over this the frame's motion is dropped and later frames rebase onto the held
    /// pose. 0 disables.
    pub covariance_gate_translation_std: f64,
    /// A teleport the covariance gate believes: m/s and rad/s against the previous raw pose.
    pub speed_gate_max_linear: f64,
    pub speed_gate_max_angular: f64,
    /// Stamp spread one frame set may span, milliseconds; 0 keeps cuVSLAM's 1 ms contract.
    pub max_skew_ms: f64,
    /// Noise of the tracker's frame-to-frame motion, [x y z roll pitch yaw], fused as a
    /// `pose_delta` source: <0 the message covariance, >0 fixed, 0 drops a translation axis
    /// and gives a rotation axis the camera's own (a feature located to one pixel turns the
    /// ray by 1/fx).
    pub visual_odom_pose_variances: [f64; 6],

    /// Frame the output is stamped in.
    pub odom_frame_id: String,
    /// The body the filter estimates: every sensor is placed relative to it by tf.
    pub output_frame_id: String,
    /// The first drives the filter between scans and levels it; samples from frames not
    /// listed are dropped.
    pub imus: Vec<ImuConfig>,
    /// Scans from frames not listed are dropped.
    pub lidars: Vec<LidarConfig>,
    /// External sources, each identified by the transform its estimates carry.
    pub odom_sources: Vec<SourceConfig>,
    pub process_noise: ProcessNoise,
}

/// Defers to the tracker so a field's default cannot drift from the core that reads it.
impl Default for DimSlamCoreConfig {
    fn default() -> Self {
        let cuvslam = CuvslamOdometryConfig::default();
        Self {
            camera_mode: cuvslam.camera_mode,
            cameras: cuvslam.cameras,
            use_gpu: cuvslam.use_gpu,
            rig_frame_id: cuvslam.rig_frame_id,
            map_frame_id: cuvslam.map_frame_id,
            covariance_gate_translation_std: cuvslam.covariance_gate_translation_std,
            speed_gate_max_linear: cuvslam.speed_gate_max_linear,
            speed_gate_max_angular: cuvslam.speed_gate_max_angular,
            max_skew_ms: cuvslam.max_skew_ms,
            visual_odom_pose_variances: [0.0; 6],
            odom_frame_id: "odom".to_string(),
            output_frame_id: "base_link".to_string(),
            imus: Vec::new(),
            lidars: Vec::new(),
            odom_sources: Vec::new(),
            process_noise: ProcessNoise::default(),
        }
    }
}

/// A source's running state: its previous message (for its steps) and, for a `pose` source,
/// where its world sits in the filter's.
#[derive(Clone, Debug, Default)]
struct SourceState {
    config: SourceConfig,
    previous: Option<(u64, i64, Isometry3<f64>)>,
    world_from_source: Option<Isometry3<f64>>,
}

/// Odometry from any set of sensors: cuVSLAM tracking, IMUs, lidars and other odometry
/// sources, fused in one filter.
///
/// Feed it frames, IMU samples, scans and other sources' estimates, then ask for output.
pub struct DimSlamCore {
    visual_odometry: CuvslamCore,
    odometry: Box<LioOdometry>,
    output_frame_id: String,
    odom_frame_id: String,
    imus: Vec<String>,
    lidars: Vec<String>,
    /// The configured sources, then the tracker.
    sources: Vec<SourceState>,
    /// rad^2: one frame's rotation variance from the camera's focal length, once known.
    focal_variance: Option<f64>,
    newest: Option<OdometryEstimate>,
}

impl DimSlamCore {
    pub fn new(config: DimSlamCoreConfig) -> Result<Self, String> {
        let visual_odometry = CuvslamCore::new(CuvslamOdometryConfig {
            camera_mode: config.camera_mode,
            cameras: config.cameras,
            use_gpu: config.use_gpu,
            odom_frame_id: VISUAL_ODOM_FRAME_ID.to_string(),
            output_frame_id: config.output_frame_id.clone(),
            rig_frame_id: config.rig_frame_id,
            map_frame_id: config.map_frame_id,
            covariance_gate_translation_std: config.covariance_gate_translation_std,
            speed_gate_max_linear: config.speed_gate_max_linear,
            speed_gate_max_angular: config.speed_gate_max_angular,
            // The filter owns the IMUs; cuVSLAM's own inertial mode would double-count them.
            enable_imu: false,
            max_skew_ms: config.max_skew_ms,
        })?;
        let mut lio = LioConfig::default();
        let noise = config.process_noise;
        lio.process[0] = noise.velocity;
        lio.process[1] = noise.angular_velocity;
        lio.process[2] = noise.specific_force;
        let imus = config
            .imus
            .iter()
            .map(|imu| {
                let mut measurement = point_lio_imu();
                measurement.gyro_variance = imu.gyro_variance;
                measurement.accel_variance = imu.accel_variance;
                measurement.clip = (imu.gyro_clip > 0.0 || imu.accel_clip_g > 0.0).then_some((
                    if imu.gyro_clip > 0.0 { imu.gyro_clip } else { f64::INFINITY },
                    if imu.accel_clip_g > 0.0 { imu.accel_clip_g } else { f64::INFINITY },
                ));
                (measurement, [imu.gyro_bias_random_walk, imu.accel_bias_random_walk])
            })
            .collect();
        let mut sources: Vec<SourceState> = config
            .odom_sources
            .into_iter()
            .map(|config| SourceState { config, ..Default::default() })
            .collect();
        sources.push(SourceState {
            config: SourceConfig {
                parent_frame_id: VISUAL_ODOM_FRAME_ID.to_string(),
                child_frame_id: config.output_frame_id.clone(),
                kind: SourceKind::PoseDelta,
                pose_variances: config.visual_odom_pose_variances,
                ..Default::default()
            },
            ..Default::default()
        });
        Ok(Self {
            visual_odometry,
            odometry: Box::new(LioOdometry::new(lio, imus, config.lidars.len())),
            output_frame_id: config.output_frame_id,
            odom_frame_id: config.odom_frame_id,
            imus: config.imus.into_iter().map(|imu| imu.frame_id).collect(),
            lidars: config.lidars.into_iter().map(|lidar| lidar.frame_id).collect(),
            sources,
            focal_variance: None,
            newest: None,
        })
    }

    pub fn handle_camera_info(&mut self, info: CameraModel, tf: &dyn TfLookup) {
        let focal = info.intrinsics[0];
        if self.focal_variance.is_none() && focal > 0.0 {
            self.focal_variance = Some(1.0 / (focal * focal));
        }
        self.visual_odometry.handle_camera_info(info, tf);
    }

    pub fn handle_depth_camera_info(&mut self, info: CameraModel) {
        self.visual_odometry.handle_depth_camera_info(info);
    }

    pub fn handle_image(&mut self, img: ImageFrame, tf: &dyn TfLookup) {
        let tracked = self.visual_odometry.handle_image(img, tf);
        let segment = self.visual_odometry.segment();
        self.handle_tracked(tracked, segment);
    }

    /// The tracker's estimate for an image without fusing it, and the tracking segment it
    /// belongs to: with `handle_tracked`, lets a caller record the tracker and replay it later.
    pub fn track_image(&mut self, img: ImageFrame, tf: &dyn TfLookup) -> (Option<OdometryEstimate>, u64) {
        let tracked = self.visual_odometry.handle_image(img, tf);
        (tracked, self.visual_odometry.segment())
    }

    /// Fuses an estimate `track_image` returned, in this run or a recorded one.
    pub fn handle_tracked(&mut self, tracked: Option<OdometryEstimate>, segment: u64) {
        // A tracker's first estimate can come back unstamped.
        let Some(estimate) = tracked.filter(|e| e.timestamp_ns != 0) else { return };
        let visual = self.sources.len() - 1;
        let mut variances = self.sources[visual].config.pose_variances;
        for variance in &mut variances[3..] {
            if *variance == 0.0 {
                *variance = self.focal_variance.unwrap_or(0.0);
            }
        }
        self.step(visual, segment, &estimate, &variances);
        self.drain();
    }

    /// Returns the range-gated depth cloud, in the depth sensor's own frame, when there is one.
    pub fn handle_depth_image(&mut self, img: ImageFrame, tf: &dyn TfLookup) -> Option<PointCloud> {
        let (cloud, tracked) = self.visual_odometry.handle_depth_image(img, tf);
        let segment = self.visual_odometry.segment();
        self.handle_tracked(tracked, segment);
        cloud
    }

    /// A sample from a listed IMU frame, placed relative to the output frame by tf.
    pub fn handle_imu(&mut self, sample: &ImuSample, tf: &dyn TfLookup) {
        let Some(imu) = self.imus.iter().position(|frame| *frame == sample.frame_id) else {
            warn_throttled!(
                std::time::Duration::from_secs(10),
                imu_frame = %sample.frame_id,
                "imu dropped: its frame is not in `imus`",
            );
            return;
        };
        if !self.odometry.imu_mounted(imu) {
            let Some(mount) = self.mount(&sample.frame_id, tf) else { return };
            self.odometry.set_imu_mount(imu, mount.0, mount.1);
        }
        self.odometry.push_imu(
            imu,
            header_seconds(sample.timestamp_ns),
            sample.angular_velocity,
            sample.linear_acceleration / STANDARD_GRAVITY,
        );
        self.drain();
    }

    /// A scan from a listed lidar frame, placed relative to the output frame by tf.
    pub fn handle_cloud(&mut self, scan: &LidarScan, tf: &dyn TfLookup) {
        let Some(lidar) = self.lidars.iter().position(|frame| *frame == scan.frame_id) else {
            warn_throttled!(
                std::time::Duration::from_secs(10),
                lidar_frame = %scan.frame_id,
                "scan dropped: its frame is not in `lidars`",
            );
            return;
        };
        if !self.odometry.lidar_mounted(lidar) {
            let Some(mount) = self.mount(&scan.frame_id, tf) else { return };
            self.odometry.set_lidar_mount(lidar, mount.0, mount.1);
        }
        self.odometry.push_scan(lidar, scan);
        self.drain();
    }

    /// An external odometry source's estimate, told apart by the transform it carries.
    pub fn handle_source(&mut self, msg: &OdometryEstimate, tf: &dyn TfLookup) {
        let Some(index) = self.sources.iter().position(|source| {
            source.config.parent_frame_id == msg.frame_id
                && source.config.child_frame_id == msg.child_frame_id
        }) else {
            warn_throttled!(
                std::time::Duration::from_secs(10),
                parent = %msg.frame_id,
                child = %msg.child_frame_id,
                "odometry dropped: no `odom_sources` entry carries this transform",
            );
            return;
        };
        let Some(child_from_output) = self.placed(&msg.child_frame_id, tf) else {
            warn_throttled!(
                std::time::Duration::from_secs(10),
                child_frame = %msg.child_frame_id,
                output_frame = %self.output_frame_id,
                "odometry dropped: tf does not place its frame relative to the output frame",
            );
            return;
        };
        let carried = OdometryEstimate {
            child_frame_id: self.output_frame_id.clone(),
            pose: msg.pose * child_from_output,
            ..msg.clone()
        };
        let config = self.sources[index].config.clone();
        let stamp = header_seconds(msg.timestamp_ns);
        if config.twist_variances.iter().any(|v| *v != 0.0) {
            self.twist(index, stamp, msg, &child_from_output);
        }
        match config.kind {
            SourceKind::Twist => {}
            SourceKind::PoseDelta => self.step(index, 0, &carried, &config.pose_variances),
            SourceKind::Pose => self.pose(index, stamp, &carried),
        }
        self.drain();
    }

    /// The newest pose the filter has published, once.
    pub fn maybe_publish(&mut self) -> Option<OdometryEstimate> {
        self.newest.take()
    }

    /// As `maybe_publish`: the filter publishes as its inputs move it, not on a timer.
    pub fn maybe_publish_at(&mut self, _now_ns: i64) -> Option<OdometryEstimate> {
        self.newest.take()
    }

    pub fn report(&self) {
        self.visual_odometry.report();
        tracing::info!(
            scans = self.odometry.scans_pushed(),
            late_points = self.odometry.late_points(),
            "odometry counters"
        );
    }

    /// output_from_frame: a sensor's mount.
    fn mount(&self, frame: &str, tf: &dyn TfLookup) -> Option<(Matrix3<f64>, Vector3<f64>)> {
        let Some(output_from_frame) = self.placed(frame, tf).map(|t| t.inverse()) else {
            warn_throttled!(
                std::time::Duration::from_secs(10),
                frame = %frame,
                output_frame = %self.output_frame_id,
                "sensor dropped until tf places its frame relative to the output frame",
            );
            return None;
        };
        if frame == self.output_frame_id {
            return Some((Matrix3::identity(), Vector3::zeros()));
        }
        Some((
            output_from_frame.rotation.to_rotation_matrix().into_inner(),
            output_from_frame.translation.vector,
        ))
    }

    /// frame_from_output, identity for the output frame itself.
    fn placed(&self, frame: &str, tf: &dyn TfLookup) -> Option<Isometry3<f64>> {
        if frame == self.output_frame_id {
            return Some(Isometry3::identity());
        }
        tf.latest(frame, &self.output_frame_id)
    }

    /// A source's twist, read in its child frame and carried to the output frame's origin.
    fn twist(&mut self, index: usize, stamp: f64, msg: &OdometryEstimate, child_from_output: &Isometry3<f64>) {
        let config = &self.sources[index].config;
        let (linear, angular) = (msg.twist.linear, msg.twist.angular);
        let to_output = child_from_output.rotation.inverse();
        if config.still_variance > 0.0
            && linear.norm() <= config.still_speed
            && angular.norm() <= config.still_angular_speed
        {
            let still = Vector3::repeat(config.still_variance);
            self.odometry.push_body_velocity(2 * index, stamp, Vector3::zeros(), still);
            self.odometry.push_rate(2 * index + 1, stamp, Vector3::zeros(), still);
            return;
        }
        let variances: [f64; 6] =
            std::array::from_fn(|i| resolve_variance(config.twist_variances[i], msg.twist_covariance[(i, i)]));
        let velocity = to_output * (linear + angular.cross(&child_from_output.translation.vector));
        self.odometry.push_body_velocity(2 * index, stamp, velocity, Vector3::new(variances[0], variances[1], variances[2]));
        self.odometry.push_rate(2 * index + 1, stamp, to_output * angular, Vector3::new(variances[3], variances[4], variances[5]));
    }

    /// A drifting pose's step since the source's previous message (within one tracking
    /// segment), as the body's velocity and angular velocity at the step's middle.
    fn step(&mut self, index: usize, segment: u64, carried: &OdometryEstimate, pose_variances: &[f64; 6]) {
        let (t, pose) = (carried.timestamp_ns, carried.pose);
        let previous = self.sources[index].previous.replace((segment, t, pose));
        let Some((_, t0, pose0)) = previous.filter(|(last_segment, ..)| *last_segment == segment) else {
            return;
        };
        let dt = (t - t0) as f64 / 1e9;
        if dt <= 0.0 {
            return;
        }
        let variances: [f64; 6] = std::array::from_fn(|i| {
            resolve_variance(pose_variances[i], carried.pose_covariance[(i, i)]) / (dt * dt)
        });
        let turn = (pose0.rotation.inverse() * pose.rotation).scaled_axis();
        let middle = pose0.rotation * UnitQuaternion::from_scaled_axis(turn / 2.0);
        let velocity = middle.inverse_transform_vector(&(pose.translation.vector - pose0.translation.vector)) / dt;
        let stamp = (t0 + t) as f64 / 2e9;
        if variances[..3].iter().any(|v| *v > 0.0) {
            self.odometry.push_body_velocity(2 * index, stamp, velocity, Vector3::new(variances[0], variances[1], variances[2]));
        }
        if variances[3..].iter().any(|v| *v > 0.0) {
            self.odometry.push_rate(2 * index + 1, stamp, turn / dt, Vector3::new(variances[3], variances[4], variances[5]));
        }
    }

    /// A non-drifting pose: its first message, once the filter runs, aligns the source's world
    /// to the filter's; later ones are measurements of the body's pose.
    fn pose(&mut self, index: usize, stamp: f64, carried: &OdometryEstimate) {
        let source = &mut self.sources[index];
        let Some(world_from_source) = source.world_from_source else {
            if self.odometry.started() {
                let x = self.odometry.state();
                let body = Isometry3::from_parts(
                    Translation3::from(x.position),
                    UnitQuaternion::from_matrix(&x.rotation),
                );
                source.world_from_source = Some(body * carried.pose.inverse());
            }
            return;
        };
        let measured = world_from_source * carried.pose;
        let variances: [f64; 6] = std::array::from_fn(|i| {
            resolve_variance(source.config.pose_variances[i], carried.pose_covariance[(i, i)])
        });
        self.odometry.push_pose(
            2 * index,
            stamp,
            measured.translation.vector,
            measured.rotation.to_rotation_matrix().into_inner(),
            variances,
        );
    }

    /// Runs the filter as far as its inputs reach, keeping the newest pose it publishes.
    fn drain(&mut self) {
        while let Some(published) = self.odometry.step() {
            let Some(pose) = published else { continue };
            self.newest = Some(estimate(&pose, &self.odom_frame_id, &self.output_frame_id));
        }
    }
}

fn estimate(pose: &LioPose, odom_frame_id: &str, output_frame_id: &str) -> OdometryEstimate {
    let [x, y, z, w] = pose.quaternion;
    let rotation = UnitQuaternion::from_quaternion(Quaternion::new(w, x, y, z));
    OdometryEstimate {
        timestamp_ns: (pose.stamp * 1e9).round() as i64,
        frame_id: odom_frame_id.to_string(),
        child_frame_id: output_frame_id.to_string(),
        pose: Isometry3::from_parts(Translation3::from(pose.position), rotation),
        twist: Twist {
            linear: rotation.inverse_transform_vector(&pose.velocity),
            angular: pose.angular_velocity,
        },
        ..Default::default()
    }
}

/// <0 takes the message's, 0 drops; a non-finite or non-positive result drops too.
fn resolve_variance(configured: f64, from_message: f64) -> f64 {
    let variance = if configured < 0.0 { from_message } else { configured };
    if variance.is_finite() && variance > 0.0 {
        variance
    } else {
        0.0
    }
}
