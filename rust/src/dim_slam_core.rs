// Copyright 2026 Dimensional Inc.
// SPDX-License-Identifier: Apache-2.0
//
// The cuVSLAM tracker is a fusion source like any other, so it is wired up here rather than by
// every caller: its world frame, its place in the source list and its `enable_imu` all have to
// agree with the filter's, and none of that is a choice a caller gets to make.

use serde::{Deserialize, Serialize};

use crate::cuvslam_odometry::{CameraConfig, CuvslamCore, CuvslamOdometryConfig};
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
    use_imu: bool,
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
        let fusion_config = OdometryFusionConfig {
            odom_frame_id: config.odom_frame_id,
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
        };
        Ok(Self {
            visual_odometry,
            output_frame_id: config.output_frame_id,
            use_imu: fusion_config.use_imu(),
            fusion: FusionCore::new(fusion_config),
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
        self.fusion.maybe_publish()
    }

    /// The fused pose predicted to `now_ns`, for a caller publishing on its own timer.
    pub fn maybe_publish_at(&mut self, now_ns: i64) -> Option<OdometryEstimate> {
        self.fusion.maybe_publish_at(now_ns)
    }

    pub fn report(&self) {
        self.visual_odometry.report();
        self.fusion.report();
    }

    fn fuse(&mut self, tracked: Option<OdometryEstimate>) {
        if let Some(estimate) = tracked {
            self.fusion.handle_source(&estimate);
        }
    }
}
