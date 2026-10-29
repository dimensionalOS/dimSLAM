// Copyright 2026 Dimensional Inc.
// SPDX-License-Identifier: Apache-2.0
//
// The cuVSLAM tracker is a fusion source like any other, so it is wired up here rather than by
// every caller: its world frame, its place in the source list and its `enable_imu` all have to
// agree with the filter's, and none of that is a choice a caller gets to make.

use serde::{Deserialize, Serialize};

use nalgebra::{Isometry3, Quaternion, Translation3, UnitQuaternion};

use crate::cuvslam_odometry::{CameraConfig, CuvslamCore, CuvslamOdometryConfig};
use crate::odometry_fusion::lio_odometry::{LioConfig, LioOdometry};
use crate::odometry_fusion::{header_seconds, STANDARD_GRAVITY};
use crate::odometry_fusion::{
    FusionCore, ImuConfig, InitialStds, LidarConfig, LidarMapConfig, OdometryFusionConfig, ProcessNoise, SourceConfig,
    SourceKind,
};
use crate::types::{
    CameraModel, ImageFrame, ImuSample, LidarScan, OdometryEstimate, PointCloud, TfLookup,
};
use crate::warn_throttled;

/// The tracker's own drifting world frame. It never leaves this struct: it exists only to tell
/// the tracker's estimates apart from the other sources inside the filter.
const VISUAL_ODOM_FRAME_ID: &str = "visual_odom";

/// Turns in a row the LIO refuses before the tracker is rebuilt (~1 s at 15 Hz): mono cuVSLAM
/// can freeze without reporting itself lost.
const REFUSED_TURNS: u32 = 15;

/// Point-LIO as the lidar-inertial filter, in place of the fusion core's; the tracker's
/// frame-to-frame turn is fused into it as a measurement of the body's angular velocity.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct PointLioEngineConfig {
    /// g: the accelerometer's clip level, when it differs from Point-LIO's assumed 3 g.
    #[serde(default)]
    pub accel_clip_g: Option<f64>,
    /// rad/s: the gyro's clip level, when it differs from Point-LIO's assumed 35.
    #[serde(default)]
    pub gyro_clip: Option<f64>,
    /// rad^2: variance of one frame-to-frame rotation, per axis.
    pub visual_rotation_variance: f64,
}

struct PointLioEngine {
    odometry: Box<LioOdometry>,
    /// The tracker's last stamp and rotation, the start of the next turn.
    last_turn: Option<(i64, UnitQuaternion<f64>)>,
    rotation_variance: f64,
    newest: Option<OdometryEstimate>,
}

impl PointLioEngine {
    fn new(config: &PointLioEngineConfig) -> Self {
        let mut lio = LioConfig::default();
        if let Some(clip) = config.accel_clip_g {
            lio.imu.clip = lio.imu.clip.map(|(gyro, _)| (gyro, clip));
        }
        if let Some(clip) = config.gyro_clip {
            lio.imu.clip = lio.imu.clip.map(|(_, accel)| (clip, accel));
        }
        Self {
            odometry: Box::new(LioOdometry::new(lio)),
            last_turn: None,
            rotation_variance: config.visual_rotation_variance,
            newest: None,
        }
    }

    /// Keeps the newest of the poses the LIO has finished.
    fn drain(&mut self, odom_frame_id: &str, output_frame_id: &str) {
        while let Some(published) = self.odometry.step() {
            let Some(pose) = published else { continue };
            let [x, y, z, w] = pose.quaternion;
            let rotation = UnitQuaternion::from_quaternion(Quaternion::new(w, x, y, z));
            self.newest = Some(OdometryEstimate {
                timestamp_ns: (pose.stamp * 1e9).round() as i64,
                frame_id: odom_frame_id.to_string(),
                child_frame_id: output_frame_id.to_string(),
                pose: Isometry3::from_parts(Translation3::from(pose.position), rotation),
                twist: crate::types::Twist {
                    linear: rotation.inverse_transform_vector(&pose.velocity),
                    angular: pose.angular_velocity,
                },
                ..Default::default()
            });
        }
    }

    /// The tracker's turn since its last frame, over the interval, as an angular-velocity
    /// measurement. False when the LIO has refused so many in a row that the tracker is stuck.
    fn turn(&mut self, tracked: &OdometryEstimate) -> bool {
        let (t, rotation) = (tracked.timestamp_ns, tracked.pose.rotation);
        if self.odometry.rates_rejected_in_a_row >= REFUSED_TURNS {
            self.odometry.rates_rejected_in_a_row = 0;
            self.last_turn = None;
            return false;
        }
        if let Some((t0, r0)) = self.last_turn {
            let dt = (t - t0) as f64 / 1e9;
            if dt > 0.0 {
                let rate = (r0.inverse() * rotation).scaled_axis() / dt;
                self.odometry
                    .push_rate((t0 + t) as f64 / 2e9, rate, self.rotation_variance / (dt * dt));
            }
        }
        self.last_turn = Some((t, rotation));
        true
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(default)]
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
    /// Noise of the tracker's frame-to-frame motion, [x y z roll pitch yaw]: it drifts, so
    /// it is fused as a `pose_delta` source. Negative takes the message covariance, zero drops
    /// the dimension, positive is a fixed variance. Its covariance describes accumulated
    /// drift rather than one frame's motion: prefer a fixed value.
    pub visual_odom_pose_variances: [f64; 6],

    /// Frame the fused output is stamped in.
    pub odom_frame_id: String,
    /// The body frame everything is resolved into: the tracker's output and the frame the
    /// filter estimates.
    pub output_frame_id: String,
    /// Output cadence; each output is the filter predicted to its own stamp.
    pub publish_rate: f64,
    /// How far back a late measurement can reach before it is dropped.
    pub replay_buffer_seconds: f64,
    /// Outlier gate: squared Mahalanobis distance per measurement dimension; 0 is no gate.
    pub outlier_rejection_allowed_variance: f64,
    /// Gate on the filter's own state, which can run away with no bad measurement; 0 disables.
    pub max_position_m: f64,
    /// Empty predicts by constant velocity instead; each entry keeps its own noise figures
    /// and biases.
    pub imus: Vec<ImuConfig>,
    /// m/s^2, seeding the filter's gravity estimate.
    pub initial_gravity_estimate: f64,
    pub initial_stds: InitialStds,
    /// The constant-velocity model's noise; only used without IMUs.
    pub process_noise: ProcessNoise,
    /// External sources, each identified by the transform its estimates carry.
    pub odom_sources: Vec<SourceConfig>,
    /// A ground body: z, roll, pitch and their rates held at zero.
    pub planar: bool,
    /// See [`OdometryFusionConfig::zero_velocity_variances`].
    pub zero_velocity_variances: [f64; 6],
    /// See [`OdometryFusionConfig::no_imu_blend`].
    pub no_imu_blend: bool,
    /// See [`OdometryFusionConfig::imm`].
    pub imm: bool,
    /// See [`OdometryFusionConfig::lidars`].
    pub lidars: Vec<LidarConfig>,
    pub lidar_map: LidarMapConfig,
    /// See [`OdometryFusionConfig::imu_as_measurement`].
    pub imu_as_measurement: bool,
    /// Set, Point-LIO is the lidar-inertial filter (fed the raw IMU, in its own frame, and
    /// the scans) and the fusion core's lidar and IMU settings are unused.
    pub point_lio: Option<PointLioEngineConfig>,
}

/// Defers to the two cores so a field's default cannot drift from the core that reads it.
impl Default for DimSlamCoreConfig {
    fn default() -> Self {
        let cuvslam = CuvslamOdometryConfig::default();
        let fusion = OdometryFusionConfig::default();
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
            odom_frame_id: fusion.odom_frame_id,
            output_frame_id: fusion.output_frame_id,
            publish_rate: fusion.publish_rate,
            replay_buffer_seconds: fusion.replay_buffer_seconds,
            outlier_rejection_allowed_variance: fusion.outlier_rejection_allowed_variance,
            max_position_m: fusion.max_position_m,
            imus: fusion.imus,
            initial_gravity_estimate: fusion.initial_gravity_estimate,
            initial_stds: fusion.initial_stds,
            process_noise: fusion.process_noise,
            odom_sources: fusion.odom_sources,
            planar: fusion.planar,
            zero_velocity_variances: fusion.zero_velocity_variances,
            no_imu_blend: fusion.no_imu_blend,
            imm: fusion.imm,
            lidars: fusion.lidars,
            lidar_map: fusion.lidar_map,
            imu_as_measurement: fusion.imu_as_measurement,
            point_lio: None,
        }
    }
}

/// Visual-inertial odometry: cuVSLAM tracking fused with any other odometry source.
///
/// Feed it frames, IMU samples and other sources' estimates, then ask for output. The tracker's
/// pose is fused in-process and never leaves the struct.
pub struct DimSlamCore {
    visual_odometry: CuvslamCore,
    fusion: FusionCore,
    output_frame_id: String,
    odom_frame_id: String,
    use_imu: bool,
    point_lio: Option<PointLioEngine>,
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
            // The filter owns the IMU; cuVSLAM's own inertial mode would double-count it.
            enable_imu: false,
            max_skew_ms: config.max_skew_ms,
        })?;
        let mut odom_sources = vec![SourceConfig {
            parent_frame_id: VISUAL_ODOM_FRAME_ID.to_string(),
            child_frame_id: config.output_frame_id.clone(),
            kind: SourceKind::PoseDelta,
            pose_variances: config.visual_odom_pose_variances,
            ..Default::default()
        }];
        odom_sources.extend(config.odom_sources);
        let point_lio = config.point_lio.as_ref().map(PointLioEngine::new);
        let fusion_config = OdometryFusionConfig {
            odom_frame_id: config.odom_frame_id.clone(),
            output_frame_id: config.output_frame_id.clone(),
            publish_rate: config.publish_rate,
            replay_buffer_seconds: config.replay_buffer_seconds,
            outlier_rejection_allowed_variance: config.outlier_rejection_allowed_variance,
            max_position_m: config.max_position_m,
            imus: config.imus,
            initial_gravity_estimate: config.initial_gravity_estimate,
            initial_stds: config.initial_stds,
            process_noise: config.process_noise,
            odom_sources,
            planar: config.planar,
            zero_velocity_variances: config.zero_velocity_variances,
            no_imu_blend: config.no_imu_blend,
            imm: config.imm,
            lidars: config.lidars,
            lidar_map: config.lidar_map,
            imu_as_measurement: config.imu_as_measurement,
        };
        Ok(Self {
            visual_odometry,
            output_frame_id: config.output_frame_id,
            odom_frame_id: config.odom_frame_id,
            use_imu: fusion_config.use_imu(),
            fusion: FusionCore::new(fusion_config),
            point_lio,
        })
    }

    pub fn handle_camera_info(&mut self, info: CameraModel, tf: &dyn TfLookup) {
        self.visual_odometry.handle_camera_info(info, tf);
    }

    pub fn handle_depth_camera_info(&mut self, info: CameraModel) {
        self.visual_odometry.handle_depth_camera_info(info);
    }

    pub fn handle_image(&mut self, img: ImageFrame, tf: &dyn TfLookup) {
        let tracked = self.visual_odometry.handle_image(img, tf);
        self.fuse(tracked);
    }

    /// Returns the range-gated depth cloud, in the depth sensor's own frame, when there is one.
    pub fn handle_depth_image(&mut self, img: ImageFrame, tf: &dyn TfLookup) -> Option<PointCloud> {
        let (cloud, tracked) = self.visual_odometry.handle_depth_image(img, tf);
        self.fuse(tracked);
        cloud
    }

    /// A no-op with no IMU configured. The mount is read off `tf`, so the sample's frame must
    /// be placed in the output frame.
    pub fn handle_imu(&mut self, sample: &ImuSample, tf: &dyn TfLookup) {
        if let Some(lio) = &mut self.point_lio {
            lio.odometry.push_imu(
                header_seconds(sample.timestamp_ns),
                sample.angular_velocity,
                sample.linear_acceleration / STANDARD_GRAVITY,
            );
            lio.drain(&self.odom_frame_id, &self.output_frame_id);
            return;
        }
        if !self.use_imu {
            return;
        }
        let Some(base_from_imu) = tf.latest(&self.output_frame_id, &sample.frame_id) else {
            warn_throttled!(
                std::time::Duration::from_secs(10),
                imu_frame = %sample.frame_id,
                "imu dropped: tf does not place the IMU in the output frame",
            );
            return;
        };
        self.fusion.handle_imu(sample, &base_from_imu);
    }

    /// A no-op for a frame with no `lidars` entry. The mount is read off `tf`, so the scan's
    /// frame must be placed in the output frame.
    pub fn handle_cloud(&mut self, scan: &LidarScan, tf: &dyn TfLookup) {
        if let Some(lio) = &mut self.point_lio {
            lio.odometry.push_scan(scan);
            lio.drain(&self.odom_frame_id, &self.output_frame_id);
            return;
        }
        let Some(base_from_lidar) = tf.latest(&self.output_frame_id, &scan.frame_id) else {
            warn_throttled!(
                std::time::Duration::from_secs(10),
                lidar_frame = %scan.frame_id,
                "scan dropped: tf does not place the lidar in the output frame",
            );
            return;
        };
        self.fusion.handle_cloud(scan, &base_from_lidar);
    }

    /// An external odometry source's estimate, told apart by the transform it carries.
    pub fn handle_source(&mut self, msg: &OdometryEstimate) {
        self.fusion.handle_source(msg);
    }

    /// The fused pose, once a publish period has elapsed.
    pub fn maybe_publish(&mut self) -> Option<OdometryEstimate> {
        if let Some(lio) = &mut self.point_lio {
            return lio.newest.take();
        }
        self.fusion.maybe_publish()
    }

    /// The fused pose predicted to `now_ns`, for a caller publishing on its own timer.
    pub fn maybe_publish_at(&mut self, now_ns: i64) -> Option<OdometryEstimate> {
        if let Some(lio) = &mut self.point_lio {
            return lio.newest.take();
        }
        self.fusion.maybe_publish_at(now_ns)
    }

    pub fn report(&self) {
        self.visual_odometry.report();
        self.fusion.report();
    }

    fn fuse(&mut self, tracked: Option<OdometryEstimate>) {
        let Some(estimate) = tracked else { return };
        // A tracker's first estimate can come back unstamped.
        match &mut self.point_lio {
            Some(lio) if estimate.timestamp_ns != 0 => {
                if !lio.turn(&estimate) {
                    warn_throttled!(
                        std::time::Duration::from_secs(10),
                        refused = REFUSED_TURNS,
                        "the LIO refused the tracker's turns: rebuilding the tracker",
                    );
                    self.visual_odometry.reset();
                }
                lio.drain(&self.odom_frame_id, &self.output_frame_id);
            }
            Some(_) => {}
            None => self.fusion.handle_source(&estimate),
        }
    }
}
