// Copyright 2026 Dimensional Inc.
// SPDX-License-Identifier: Apache-2.0
//
// cuVSLAM visual odometry enters the fusion filter in-process as a drifting source, so its
// pose never touches the wire.

mod cuvslam_odometry;
mod odometry_fusion;

use cuvslam_odometry::imu_info::ImuInfo;
use cuvslam_odometry::{CuvslamOdometryConfig, VoCore};
use dimos_module::{
    native_config, run_with_transport, warn_throttled, Input, Module, Output, Tf,
};
use lcm_msgs::nav_msgs::Odometry;
use lcm_msgs::sensor_msgs::{CameraInfo, Image, Imu, PointCloud2};
use odometry_fusion::{FusionCore, OdometryFusionConfig};

#[native_config]
struct DimSlamConfig {
    camera_mode: String,
    camera_frames: Vec<String>,
    rectified: bool,
    use_gpu: bool,
    /// Frame the tracker's internal odometry drifts in; fused as a drifting source,
    /// so it must appear in source_frames.
    visual_odom_frame: String,
    rig_frame: String,
    covariance_gate_translation_std: f64,
    speed_gate_max_linear: f64,
    speed_gate_max_angular: f64,
    /// cuVSLAM's own inertial mode, separate from the filter's use_imu.
    cuvslam_enable_imu: bool,
    depth_units_per_meter: f64,
    depth_cloud_min_range: f64,
    depth_cloud_max_range: f64,
    depth_cloud_decimation: i64,

    odom_frame: String,
    base_frame: String,
    publish_tf: bool,
    publish_rate: f64,
    replay_buffer_seconds: f64,
    mahalanobis_gate: f64,
    max_position_m: f64,
    use_imu: bool,
    imu_gyro_noise_density: f64,
    imu_gyro_random_walk: f64,
    imu_accel_noise_density: f64,
    imu_accel_random_walk: f64,
    gravity: f64,
    imu_init_samples: i64,
    initial_position_std: f64,
    initial_velocity_std: f64,
    initial_rotation_std: f64,
    initial_bias_std: f64,
    source_frames: Vec<String>,
    source_pose_variances: Vec<f64>,
    source_twist_variances: Vec<f64>,
    constraint_twist_variances: Vec<f64>,
}

#[derive(Module)]
#[module(setup = init, teardown = report)]
struct DimSlam {
    #[input(decode = CameraInfo::decode)]
    camera_info: Input<CameraInfo>,
    #[input(decode = Image::decode)]
    image: Input<Image>,
    /// rgbd only; unconnected otherwise.
    #[input(decode = Image::decode)]
    depth_image: Input<Image>,
    /// Only needed when depth has to be reprojected onto the rig camera.
    #[input(decode = CameraInfo::decode)]
    depth_camera_info: Input<CameraInfo>,
    #[input(decode = Imu::decode)]
    imu: Input<Imu>,
    /// cuvslam_enable_imu only: the IMU's noise model and frame.
    #[input(decode = ImuInfo::decode)]
    imu_info: Input<ImuInfo>,
    /// External odometry sources, told apart by header.frame_id.
    #[input(decode = Odometry::decode)]
    sources: Input<Odometry>,
    #[output(encode = Odometry::encode)]
    odometry: Output<Odometry>,
    /// rgbd only: the depth sensor's own points, range-gated, in the depth frame.
    #[output(encode = PointCloud2::encode)]
    depth_cloud: Output<PointCloud2>,
    #[tf]
    tf: Tf,
    #[config]
    config: DimSlamConfig,

    vo: Option<VoCore>,
    fusion: Option<FusionCore>,
}

impl DimSlam {
    async fn init(&mut self) {
        self.vo = Some(VoCore::new(CuvslamOdometryConfig {
            camera_mode: self.config.camera_mode.clone(),
            camera_frames: self.config.camera_frames.clone(),
            rectified: self.config.rectified,
            use_gpu: self.config.use_gpu,
            odom_frame: self.config.visual_odom_frame.clone(),
            base_frame: self.config.base_frame.clone(),
            rig_frame: self.config.rig_frame.clone(),
            covariance_gate_translation_std: self.config.covariance_gate_translation_std,
            speed_gate_max_linear: self.config.speed_gate_max_linear,
            speed_gate_max_angular: self.config.speed_gate_max_angular,
            enable_imu: self.config.cuvslam_enable_imu,
            depth_units_per_meter: self.config.depth_units_per_meter,
            depth_cloud_min_range: self.config.depth_cloud_min_range,
            depth_cloud_max_range: self.config.depth_cloud_max_range,
            depth_cloud_decimation: self.config.depth_cloud_decimation,
        }));
        self.fusion = Some(FusionCore::new(OdometryFusionConfig {
            odom_frame: self.config.odom_frame.clone(),
            base_frame: self.config.base_frame.clone(),
            publish_rate: self.config.publish_rate,
            replay_buffer_seconds: self.config.replay_buffer_seconds,
            mahalanobis_gate: self.config.mahalanobis_gate,
            max_position_m: self.config.max_position_m,
            use_imu: self.config.use_imu,
            imu_gyro_noise_density: self.config.imu_gyro_noise_density,
            imu_gyro_random_walk: self.config.imu_gyro_random_walk,
            imu_accel_noise_density: self.config.imu_accel_noise_density,
            imu_accel_random_walk: self.config.imu_accel_random_walk,
            gravity: self.config.gravity,
            imu_init_samples: self.config.imu_init_samples,
            initial_position_std: self.config.initial_position_std,
            initial_velocity_std: self.config.initial_velocity_std,
            initial_rotation_std: self.config.initial_rotation_std,
            initial_bias_std: self.config.initial_bias_std,
            source_frames: self.config.source_frames.clone(),
            source_pose_variances: self.config.source_pose_variances.clone(),
            source_twist_variances: self.config.source_twist_variances.clone(),
            constraint_twist_variances: self.config.constraint_twist_variances.clone(),
        }));
    }

    async fn handle_camera_info(&mut self, info: CameraInfo) {
        self.vo.as_mut().expect("setup ran").handle_camera_info(info, &self.tf);
    }

    async fn handle_image(&mut self, img: Image) {
        let tracked = self.vo.as_mut().expect("setup ran").handle_image(img, &self.tf);
        self.fuse(tracked).await;
    }

    async fn handle_depth_image(&mut self, img: Image) {
        let (cloud, tracked) = self.vo.as_mut().expect("setup ran").handle_depth_image(img, &self.tf);
        if let Some(cloud) = cloud {
            self.depth_cloud.publish(&cloud).await.ok();
        }
        self.fuse(tracked).await;
    }

    async fn handle_depth_camera_info(&mut self, info: CameraInfo) {
        self.vo.as_mut().expect("setup ran").handle_depth_camera_info(info);
    }

    async fn handle_imu(&mut self, msg: Imu) {
        if self.config.cuvslam_enable_imu {
            self.vo.as_mut().expect("setup ran").handle_imu(&msg);
        }
        if !self.config.use_imu {
            return;
        }
        let Some(base_from_imu) = self
            .tf
            .get_latest(&self.config.base_frame, &msg.header.frame_id)
        else {
            warn_throttled!(
                std::time::Duration::from_secs(10),
                imu_frame = %msg.header.frame_id,
                "imu dropped: tf does not place the IMU in the base frame",
            );
            return;
        };
        self.fusion.as_mut().expect("setup ran").handle_imu(&msg, &base_from_imu);
        self.publish().await;
    }

    async fn handle_imu_info(&mut self, info: ImuInfo) {
        self.vo.as_mut().expect("setup ran").handle_imu_info(info, &self.tf);
    }

    async fn handle_sources(&mut self, msg: Odometry) {
        self.fusion.as_mut().expect("setup ran").handle_source(&msg);
        self.publish().await;
    }

    async fn fuse(&mut self, tracked: Option<Odometry>) {
        let Some(visual_odometry) = tracked else { return };
        self.fusion.as_mut().expect("setup ran").handle_source(&visual_odometry);
        self.publish().await;
    }

    async fn publish(&mut self) {
        let Some((msg, odom_from_base)) = self.fusion.as_mut().expect("setup ran").maybe_publish()
        else {
            return;
        };
        self.odometry.publish(&msg).await.ok();
        if !self.config.publish_tf {
            return;
        }
        self.tf.publish(&[odom_from_base]).await.ok();
    }

    async fn report(&mut self) {
        if let Some(vo) = &self.vo {
            vo.report();
        }
        if let Some(fusion) = &self.fusion {
            fusion.report();
        }
    }
}

#[tokio::main]
async fn main() {
    run_with_transport::<DimSlam>().await;
}
