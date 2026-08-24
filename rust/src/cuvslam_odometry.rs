// Copyright 2026 Dimensional Inc.
// SPDX-License-Identifier: Apache-2.0
//
// cuVSLAM keeps one world frame for the tracker's life, so tracking loss resumes in the
// same frame; nothing is emitted while lost.

mod depth_cloud;
mod depth_reproject;
mod fused_depth;
pub mod imu_info;
mod msg_convert;

use std::collections::{HashMap, VecDeque};
use std::time::{Duration, Instant};

use dimos_module::nalgebra::{Isometry3, Matrix3, Matrix6, Vector3};
use dimos_module::{error_throttled, native_config, warn_throttled, Tf};
use lcm_msgs::nav_msgs::Odometry;
use lcm_msgs::sensor_msgs::{CameraInfo, Image, Imu, PointCloud2};
use tracing::{error, info, warn};

use self::depth_cloud::depth_cloud;
use self::depth_reproject::reproject_depth;
use self::imu_info::ImuInfo;
use self::msg_convert::{
    cuv_pose_to_isometry, stamp_to_ns, to_cuv_pose, to_distortion, to_stamp, transform_to_isometry,
};
use cu_vslam_rs::{ffi, CameraParams, ImageRef, Tracker};

/// cuVSLAM's Track() contract asks for stereo stamps within 1 ms.
const MAX_PAIR_SKEW_NS: i64 = 1_000_000;

/// A tracking stall long enough to overflow this has already outlived any use the
/// buffered inertial data had.
const MAX_PENDING_IMU: usize = 2048;

/// Colors waiting for a depth frame to pair with. The stamp cutoff normally keeps this
/// far smaller; this only bounds it when depth stops arriving.
const MAX_FUSION_COLORS: usize = 64;

#[derive(Clone, Copy, PartialEq)]
enum Mode {
    Stereo,
    Mono,
    Rgbd,
}

/// First order in the rotation block, where fixed-axis rpy rates and rotation-vector
/// components coincide.
fn se3_adjoint(parent_from_child: &Isometry3<f64>) -> Matrix6<f64> {
    let rotation = parent_from_child.rotation.to_rotation_matrix().into_inner();
    let translation = parent_from_child.translation.vector;
    let translation_skew = Matrix3::new(
        0.0,
        -translation.z,
        translation.y,
        translation.z,
        0.0,
        -translation.x,
        -translation.y,
        translation.x,
        0.0,
    );
    let mut adjoint = Matrix6::zeros();
    adjoint.fixed_view_mut::<3, 3>(0, 0).copy_from(&rotation);
    adjoint
        .fixed_view_mut::<3, 3>(0, 3)
        .copy_from(&(translation_skew * rotation));
    adjoint.fixed_view_mut::<3, 3>(3, 3).copy_from(&rotation);
    adjoint
}

/// cuVSLAM wants camera 0 to be the left of a pair, which is the one whose
/// partner sits at +x (optical convention, x to the right).
fn needs_left_swap(first_rig_from_camera: &Isometry3<f64>, second_rig_from_camera: &Isometry3<f64>) -> bool {
    (first_rig_from_camera.inverse() * second_rig_from_camera).translation.x < 0.0
}

fn odometry_mode(mode: Mode, enable_imu: bool) -> u8 {
    match mode {
        Mode::Rgbd => ffi::CUV_ODOMETRY_RGBD,
        Mode::Mono => ffi::CUV_ODOMETRY_MONO,
        // Inertial is the stereo pair plus an IMU; there is no inertial mono or rgbd.
        Mode::Stereo => {
            if enable_imu {
                ffi::CUV_ODOMETRY_INERTIAL
            } else {
                ffi::CUV_ODOMETRY_MULTICAMERA
            }
        }
    }
}

/// A three-channel image has to be declared as such: fed as MONO, cuVSLAM
/// reads a third of each row and tracks nothing.
fn image_encoding(encoding: &str) -> u8 {
    if encoding == "mono8" {
        ffi::CUV_ENCODING_MONO
    } else {
        ffi::CUV_ENCODING_RGB
    }
}

#[native_config]
#[derive(Clone)]
pub struct CuvslamOdometryConfig {
    /// "stereo", "mono" or "rgbd". Mono is accurate only up to scale.
    pub camera_mode: String,
    /// One tf frame per camera, in cuVSLAM's index order. Empty discovers them off
    /// camera_info.
    pub camera_frames: Vec<String>,
    pub rectified: bool,
    /// Needs a libcuvslam built with ENFORCE_GPU=OFF; stock SDK binaries are GPU-only.
    pub use_gpu: bool,
    /// Frame stamped on the emitted odometry; the tracker's world, drifting freely.
    pub odom_frame: String,
    pub base_frame: String,
    /// Empty means base_frame. Set it to a camera's optical frame to match NVIDIA's
    /// examples, whose rig IS the left camera.
    pub rig_frame: String,
    /// Meters; over this the frame's motion is dropped and later frames rebase onto the
    /// held pose. 0 disables.
    pub covariance_gate_translation_std: f64,
    /// A confident-covariance teleport still cannot exceed platform kinematics. Linear
    /// m/s, angular rad/s, against the previous raw pose; 0 disables.
    pub speed_gate_max_linear: f64,
    pub speed_gate_max_angular: f64,
    /// cuVSLAM's Inertial mode is stereo plus one IMU. The noise model and frame come
    /// from the imu_info stream, published by the driver the way camera_info is.
    pub enable_imu: bool,
    /// Raw depth units per metre. 1000 for sixteen-bit millimetres.
    pub depth_units_per_meter: f64,
    /// Range gate on the published depth_cloud, metres; 0 leaves it open.
    pub depth_cloud_min_range: f64,
    pub depth_cloud_max_range: f64,
    /// One median point per k x k depth block; <= 1 is off.
    pub depth_cloud_decimation: i64,
    /// RGB-guided densification of the depth image before the cloud is cut
    /// from it. Both safetensors paths set turns it on; needs the `depth2depth` cargo
    /// feature (`depth2depth-cuda`/`-cudnn`/`-metal` for a GPU).
    pub depth2depth_dinov2_weights: String,
    pub depth2depth_head_weights: String,
    /// Scales the model input resolution; 0.5 is ~4x faster and coarser.
    pub depth2depth_quality: f64,
    /// Frame whose images on the ``image`` stream feed the model; empty uses the rig
    /// camera on the depth frame. Set it when depth is aligned to a sensor with no
    /// color of its own — the few cm of parallax is small against the model's own error.
    pub depth2depth_color_frame: String,
    /// A depth image fuses with the color image closest in stamp; outside this window it
    /// gets a raw (undensified) cloud, and staying outside it for 10x this long after
    /// fusion has started gives up on densification for the rest of the run.
    pub depth2depth_max_color_skew_seconds: f64,
}

struct RigCamera {
    frame: String,
    rig_from_camera: Isometry3<f64>,
    info: CameraInfo,
    image: Option<Image>,
}

enum DepthChoice {
    Passthrough,
    Aligned,
}

pub struct VoCore {
    config: CuvslamOdometryConfig,
    /// The rig, in cuVSLAM's camera order.
    cameras: Vec<RigCamera>,
    camera_info_by_frame: HashMap<String, CameraInfo>,
    imu_model: Option<ImuInfo>,
    /// cuVSLAM needs Track() and RegisterImuMeasurement() in non-decreasing stamp order,
    /// but the dispatcher lets images overtake the IMU.
    pending_imu: VecDeque<ffi::CuvImuMeasurement>,
    vslam: Option<Tracker>,
    tracker_unbuildable: bool,

    depth: Option<Image>,
    depth_info: Option<CameraInfo>,
    depth_fuser: Option<fused_depth::Fuser>,
    /// After an ingest stall (model load, boot flood) the per-topic queues drain
    /// unevenly, so the last color seen can be seconds from the depth being processed.
    fusion_colors: VecDeque<Image>,
    fused_once: bool,
    color_stale_since: Option<Instant>,
    aligned_depth: Image,
    camera_from_depth: Option<Isometry3<f64>>,
    last_ts_ns: Option<i64>,

    world_from_base: Option<Isometry3<f64>>,
    base_from_rig: Option<Isometry3<f64>>,
    rig_from_base: Option<Isometry3<f64>>,
    /// Adjoint of base_from_rig, fixed with it at tracker creation.
    covariance_adjoint: Matrix6<f64>,
    /// identity until the gate first fires
    rebase: Option<Isometry3<f64>>,
    covariance: Matrix6<f64>,
    previous_raw: Option<Isometry3<f64>>,
    previous_raw_ns: i64,
    was_tracking: bool,

    frames: u64,
    tracked: u64,
    segment_id: u64,
    imu_samples: u64,
    imu_dropped: u64,
    skew_rejects: u64,
    depth_reprojected: u64,
    covariance_gated: u64,
    speed_gated: u64,
    unplaced_images: u64,
}

impl VoCore {
    pub fn new(config: CuvslamOdometryConfig) -> Self {
        let depth_fuser = fused_depth::Fuser::new(
            &config.depth2depth_dinov2_weights,
            &config.depth2depth_head_weights,
            config.depth2depth_quality,
        );
        Self {
            config,
            cameras: Vec::new(),
            camera_info_by_frame: HashMap::new(),
            imu_model: None,
            pending_imu: VecDeque::new(),
            vslam: None,
            tracker_unbuildable: false,
            depth: None,
            depth_info: None,
            depth_fuser,
            fusion_colors: VecDeque::new(),
            fused_once: false,
            color_stale_since: None,
            aligned_depth: Image::default(),
            camera_from_depth: None,
            last_ts_ns: None,
            world_from_base: None,
            base_from_rig: None,
            rig_from_base: None,
            covariance_adjoint: Matrix6::zeros(),
            rebase: None,
            covariance: Matrix6::zeros(),
            previous_raw: None,
            previous_raw_ns: 0,
            was_tracking: false,
            frames: 0,
            tracked: 0,
            segment_id: 0,
            imu_samples: 0,
            imu_dropped: 0,
            skew_rejects: 0,
            depth_reprojected: 0,
            covariance_gated: 0,
            speed_gated: 0,
            unplaced_images: 0,
        }
    }

    fn mode(&self) -> Mode {
        match self.config.camera_mode.as_str() {
            "stereo" => Mode::Stereo,
            "rgbd" => Mode::Rgbd,
            _ => Mode::Mono,
        }
    }

    fn rig_frame(&self) -> &str {
        if self.config.rig_frame.is_empty() {
            &self.config.base_frame
        } else {
            &self.config.rig_frame
        }
    }

    pub fn handle_camera_info(&mut self, info: CameraInfo, tf: &Tf) {
        if self.vslam.is_some() {
            return; // rig is fixed once the tracker exists
        }
        self.camera_info_by_frame.insert(info.header.frame_id.clone(), info);
        self.resolve_rig(tf);
    }

    /// All-or-nothing: no camera is placed until every camera resolves.
    fn resolve_rig(&mut self, tf: &Tf) {
        if !self.cameras.is_empty() {
            return;
        }
        let discovered = self.config.camera_frames.is_empty();
        let mut frames = self.config.camera_frames.clone();
        if discovered {
            frames.extend(self.camera_info_by_frame.keys().cloned());
            // camera_info has no order of its own, and the rig is indexed.
            frames.sort();
            let expected = if self.mode() == Mode::Stereo { 2 } else { 1 };
            if frames.len() != expected {
                if frames.len() > expected {
                    error_throttled!(
                        Duration::from_secs(10),
                        found = frames.join(", "),
                        mode = %self.config.camera_mode,
                        "cuvslam found more cameras on camera_info than camera_mode uses, \
                         and they have no discoverable order. Set camera_frames.",
                    );
                }
                return;
            }
        }

        let mut cameras = Vec::new();
        for frame in &frames {
            let Some(rig_from_camera) = tf.get_latest(self.rig_frame(), frame) else {
                return;
            };
            let Some(info) = self.camera_info_by_frame.get(frame) else {
                return;
            };
            cameras.push(RigCamera {
                frame: frame.clone(),
                rig_from_camera: transform_to_isometry(&rig_from_camera),
                info: info.clone(),
                image: None,
            });
        }
        if discovered
            && cameras.len() == 2
            && needs_left_swap(&cameras[0].rig_from_camera, &cameras[1].rig_from_camera)
        {
            cameras.swap(0, 1);
        }
        self.cameras = cameras;
    }

    fn camera_index(&self, frame_id: &str) -> Option<usize> {
        self.cameras.iter().position(|camera| camera.frame == frame_id)
    }

    pub fn handle_imu_info(&mut self, info: ImuInfo, tf: &Tf) {
        if self.vslam.is_some() {
            return; // rig is fixed once the tracker exists
        }
        self.imu_model = Some(info);
        self.resolve_rig(tf);
    }

    pub fn handle_imu(&mut self, msg: &Imu) {
        if self.vslam.is_none() {
            // No tracker yet; this is the window inertial init needs.
            self.imu_dropped += 1;
            return;
        }
        // Track() has already consumed everything up to last_ts_ns.
        self.pending_imu.push_back(ffi::CuvImuMeasurement {
            timestamp_ns: stamp_to_ns(&msg.header),
            linear_accelerations: [
                msg.linear_acceleration.x as f32,
                msg.linear_acceleration.y as f32,
                msg.linear_acceleration.z as f32,
            ],
            angular_velocities: [
                msg.angular_velocity.x as f32,
                msg.angular_velocity.y as f32,
                msg.angular_velocity.z as f32,
            ],
        });
        while self.pending_imu.len() > MAX_PENDING_IMU {
            self.pending_imu.pop_front();
        }
        self.imu_samples += 1;
    }

    pub fn handle_image(&mut self, img: Image, tf: &Tf) -> Option<Odometry> {
        if self.cameras.is_empty() {
            self.resolve_rig(tf);
        }
        let for_fusion = self.depth_fuser.is_some()
            && !self.config.depth2depth_color_frame.is_empty()
            && img.header.frame_id == self.config.depth2depth_color_frame;
        let Some(index) = self.camera_index(&img.header.frame_id) else {
            if for_fusion {
                self.push_fusion_color(img);
                return None;
            }
            self.unplaced_images += 1;
            warn_throttled!(
                Duration::from_secs(10),
                frame_id = %img.header.frame_id,
                dropped = self.unplaced_images,
                rig_cameras = self.cameras.len(),
                "cuvslam dropping image with a frame_id not on the rig",
            );
            return None;
        };
        if for_fusion {
            self.push_fusion_color(img.clone());
        }
        self.cameras[index].image = Some(img);
        self.try_track(tf)
    }

    fn push_fusion_color(&mut self, img: Image) {
        self.fusion_colors.push_back(img);
        if self.fusion_colors.len() > MAX_FUSION_COLORS {
            self.fusion_colors.pop_front();
        }
    }

    pub fn handle_depth_image(
        &mut self,
        img: Image,
        tf: &Tf,
    ) -> (Option<PointCloud2>, Option<Odometry>) {
        // Everything downstream walks the buffer by the header's step and height, so a
        // driver that undersizes it would index out of bounds and take the process down.
        let expected_bytes = img.step as usize * img.height as usize;
        if img.step < img.width * 2 || img.data.len() < expected_bytes {
            warn_throttled!(
                Duration::from_secs(10),
                width = img.width,
                height = img.height,
                step = img.step,
                bytes = img.data.len(),
                "cuvslam dropping a depth image smaller than its header claims",
            );
            return (None, None);
        }
        let cloud = self.depth_cloud_msg(&img);
        self.depth = Some(img);
        (cloud, self.try_track(tf))
    }

    fn depth_cloud_msg(&mut self, depth: &Image) -> Option<PointCloud2> {
        let depth_ns = stamp_to_ns(&depth.header);
        let limit_seconds = self.config.depth2depth_max_color_skew_seconds;
        // Colors too old for this depth frame can never match a later one either.
        let cutoff_ns = depth_ns - (limit_seconds * 1e9) as i64;
        while self
            .fusion_colors
            .front()
            .is_some_and(|color| stamp_to_ns(&color.header) < cutoff_ns)
        {
            self.fusion_colors.pop_front();
        }
        let color = if self.config.depth2depth_color_frame.is_empty() {
            self.cameras
                .iter()
                .find(|camera| camera.frame == depth.header.frame_id)
                .and_then(|camera| camera.image.as_ref())
        } else {
            self.fusion_colors
                .iter()
                .min_by_key(|color| (stamp_to_ns(&color.header) - depth_ns).abs())
        };
        let skew_seconds = color
            .map(|color| (stamp_to_ns(&color.header) - depth_ns).abs() as f64 / 1e9)
            .unwrap_or(f64::INFINITY);
        let mut give_up = false;
        let fused = match (self.depth_fuser.as_mut(), color) {
            (Some(fuser), Some(color)) if skew_seconds <= limit_seconds => {
                self.fused_once = true;
                self.color_stale_since = None;
                fuser.fuse(color, depth, self.config.depth_units_per_meter)
            }
            (Some(_), _) => {
                // A boot flood drains its backlog in well under this window, so staying
                // out of sync this long means the color stream genuinely stopped.
                let stale_since = self.color_stale_since.get_or_insert_with(Instant::now);
                give_up = self.fused_once
                    && stale_since.elapsed().as_secs_f64() > 10.0 * limit_seconds;
                if !give_up {
                    warn_throttled!(
                        Duration::from_secs(10),
                        skew_seconds,
                        depth_frame = %depth.header.frame_id,
                        "depth2depth has no color within the skew limit; publishing raw depth",
                    );
                }
                None
            }
            (None, _) => None,
        };
        if give_up {
            error!(
                limit_seconds,
                "depth2depth color stream stalled; densification is off for the rest of \
                 this run, depth clouds stay raw",
            );
            self.depth_fuser = None;
            self.fusion_colors.clear();
            self.color_stale_since = None;
        }
        let depth = fused.as_ref().unwrap_or(depth);
        let info = self.depth_info.as_ref().or_else(|| {
            self.cameras
                .iter()
                .find(|camera| camera.frame == depth.header.frame_id)
                .map(|camera| &camera.info)
        })?;
        let mut cloud = PointCloud2::default();
        depth_cloud(
            depth,
            info,
            self.config.depth_units_per_meter,
            self.config.depth_cloud_min_range,
            self.config.depth_cloud_max_range,
            self.config.depth_cloud_decimation.max(0) as u32,
            &mut cloud,
        );
        Some(cloud)
    }

    pub fn handle_depth_camera_info(&mut self, info: CameraInfo) {
        if self.depth_info.is_none() {
            self.depth_info = Some(info);
        }
    }

    /// cuVSLAM's RGBD contract needs depth pixel-aligned with the rig camera.
    fn align_depth(&mut self, tf: &Tf) -> Option<DepthChoice> {
        let depth = self.depth.as_ref().expect("checked by try_track");
        let camera = &self.cameras[0];
        if depth.header.frame_id == camera.frame {
            return Some(DepthChoice::Passthrough);
        }
        let Some(depth_info) = self.depth_info.as_ref() else {
            warn_throttled!(
                Duration::from_secs(10),
                depth_frame = %depth.header.frame_id,
                camera_frame = %camera.frame,
                "cuvslam: depth is in another frame than the camera and needs \
                 depth_camera_info to reproject",
            );
            return None;
        };
        if self.camera_from_depth.is_none() {
            let Some(camera_from_depth) = tf.get_latest(&camera.frame, &depth.header.frame_id) else {
                warn_throttled!(
                    Duration::from_secs(10),
                    depth_frame = %depth.header.frame_id,
                    camera_frame = %camera.frame,
                    "cuvslam: tf does not connect the depth frame to the camera",
                );
                return None;
            };
            self.camera_from_depth = Some(transform_to_isometry(&camera_from_depth));
            info!(
                depth_frame = %depth.header.frame_id,
                camera_frame = %camera.frame,
                "cuvslam reprojecting depth onto the rig camera"
            );
        }
        let mut aligned = std::mem::take(&mut self.aligned_depth);
        reproject_depth(
            depth,
            depth_info,
            &camera.info,
            self.camera_from_depth.as_ref().expect("set above"),
            self.config.depth_units_per_meter,
            &mut aligned,
        );
        self.aligned_depth = aligned;
        self.depth_reprojected += 1;
        Some(DepthChoice::Aligned)
    }

    fn ensure_tracker(&mut self, tf: &Tf) {
        if self.vslam.is_some() || self.tracker_unbuildable {
            return;
        }
        let Some(base_from_rig) = tf.get_latest(&self.config.base_frame, self.rig_frame()) else {
            warn_throttled!(
                Duration::from_secs(10),
                rig_frame = self.rig_frame(),
                base_frame = %self.config.base_frame,
                "cuvslam: tf does not place the rig frame against base_frame",
            );
            return;
        };
        let base_from_rig = transform_to_isometry(&base_from_rig);
        self.base_from_rig = Some(base_from_rig);
        self.rig_from_base = Some(base_from_rig.inverse());
        self.covariance_adjoint = se3_adjoint(&base_from_rig);

        let cameras: Vec<CameraParams> = self
            .cameras
            .iter()
            .map(|rig_camera| {
                let info = &rig_camera.info;
                let (distortion_model, distortion_parameters) = to_distortion(info);
                CameraParams {
                    width: info.width,
                    height: info.height,
                    principal: [info.K[2] as f32, info.K[5] as f32],
                    focal: [info.K[0] as f32, info.K[4] as f32],
                    rig_from_camera: to_cuv_pose(&rig_camera.rig_from_camera),
                    distortion_model,
                    distortion_parameters,
                }
            })
            .collect();

        let mut imu_calibration = None;
        if self.config.enable_imu {
            let Some(imu_model) = &self.imu_model else {
                warn_throttled!(
                    Duration::from_secs(10),
                    "cuvslam: enable_imu is on but no imu_info has arrived",
                );
                return;
            };
            let imu_frame = &imu_model.header.frame_id;
            let Some(rig_from_imu) = tf.get_latest(self.rig_frame(), imu_frame) else {
                warn_throttled!(
                    Duration::from_secs(10),
                    imu_frame = %imu_frame,
                    "cuvslam: enable_imu is on but tf does not place the IMU",
                );
                return;
            };
            imu_calibration = Some(ffi::CuvImuCalibration {
                rig_from_imu: to_cuv_pose(&transform_to_isometry(&rig_from_imu)),
                gyroscope_noise_density: imu_model.gyro_noise_density as f32,
                gyroscope_random_walk: imu_model.gyro_random_walk as f32,
                accelerometer_noise_density: imu_model.accel_noise_density as f32,
                accelerometer_random_walk: imu_model.accel_random_walk as f32,
                frequency: imu_model.frequency as f32,
            });
        }

        let mode = self.mode();
        let mut tracker_config = ffi::CuvConfig {
            odometry_mode: odometry_mode(mode, self.config.enable_imu),
            use_gpu: self.config.use_gpu,
            // cuVSLAM rejects this outright unless the rig has a stereo pair: "Rectified
            // stereo camera mode only works with 1+ stereo cameras".
            rectified_stereo_camera: self.config.rectified && mode == Mode::Stereo,
            rgbd_depth_scale_factor: 1.0,
            rgbd_depth_camera_id: -1,
        };
        if mode == Mode::Rgbd {
            tracker_config.rgbd_depth_scale_factor = self.config.depth_units_per_meter as f32;
            // align_depth hands every frame over in cameras[0]'s frame. The -1 default means
            // no camera, and depth belonging to nothing is silently ignored.
            tracker_config.rgbd_depth_camera_id = 0;
        }

        // Each libcuvslam build carries one backend, GPU or CPU, so a construction failure
        // with a well-formed rig means this build has the other one. Retrying the same
        // config would run vision-less forever while the fusion filter keeps publishing.
        let vslam = match Tracker::new(&cameras, imu_calibration.as_ref(), &tracker_config) {
            Ok(vslam) => vslam,
            Err(message) => {
                let fallback_config = ffi::CuvConfig {
                    use_gpu: !tracker_config.use_gpu,
                    ..tracker_config
                };
                match Tracker::new(&cameras, imu_calibration.as_ref(), &fallback_config) {
                    Ok(vslam) => {
                        warn!(
                            configured_use_gpu = tracker_config.use_gpu,
                            error = %message,
                            "cuvslam tracker construction failed with the configured \
                             backend; this build only carries the other one, using it"
                        );
                        vslam
                    }
                    Err(fallback_message) => {
                        // Nothing about the rig changes later, so a retry would only
                        // repeat this. Give up on vision and leave the rest of the
                        // process, fusion filter included, running.
                        self.tracker_unbuildable = true;
                        error!(
                            configured_use_gpu = tracker_config.use_gpu,
                            configured_error = %message,
                            fallback_error = %fallback_message,
                            "cuvslam tracker construction failed on both backends; \
                             this module will not publish odometry"
                        );
                        return;
                    }
                }
            }
        };
        self.vslam = Some(vslam);
        info!(
            cameras = cameras.len(),
            width = cameras[0].width,
            height = cameras[0].height,
            rig_frames = self
                .cameras
                .iter()
                .map(|camera| camera.frame.as_str())
                .collect::<Vec<_>>()
                .join(", "),
            "cuvslam tracker created"
        );
    }

    fn clear_frame_set(&mut self) {
        for camera in &mut self.cameras {
            camera.image = None;
        }
        self.depth = None;
    }

    fn try_track(&mut self, tf: &Tf) -> Option<Odometry> {
        let mode = self.mode();
        if self.cameras.is_empty() || (mode == Mode::Rgbd && self.depth.is_none()) {
            return None;
        }
        if self.cameras.iter().any(|camera| camera.image.is_none()) {
            return None;
        }
        self.ensure_tracker(tf);
        self.vslam.as_ref()?; // no tf placement yet

        let stamps: Vec<i64> = self
            .cameras
            .iter()
            .map(|camera| stamp_to_ns(&camera.image.as_ref().expect("checked above").header))
            .collect();
        let oldest = *stamps.iter().min().expect("nonempty");
        let newest = *stamps.iter().max().expect("nonempty");
        if newest - oldest > MAX_PAIR_SKEW_NS {
            self.skew_rejects += 1;
            warn_throttled!(
                Duration::from_secs(10),
                rejected = self.skew_rejects,
                skew_ms = (newest - oldest) as f64 / 1.0e6,
                "cuvslam frame sets exceed the 1 ms skew limit",
            );
            return None;
        }
        // cuVSLAM rejects a frame that is not strictly newer than the last one.
        if self.last_ts_ns.is_some_and(|last| newest <= last) {
            self.clear_frame_set();
            return None;
        }

        let depth_choice = if mode == Mode::Rgbd {
            // Reprojection inputs still missing means retry on the next message.
            Some(self.align_depth(tf)?)
        } else {
            None
        };

        let vslam = self.vslam.as_mut().expect("checked above");
        // Popping the front rather than partitioning: one late sample would make the deque
        // unsorted, and partition_point would then silently drop the whole backlog.
        while self.pending_imu.front().is_some_and(|m| m.timestamp_ns <= newest) {
            let measurement = self.pending_imu.pop_front().expect("checked above");
            if let Err(message) = vslam.register_imu(&measurement) {
                warn_throttled!(Duration::from_secs(10), error = %message, "cuvslam rejected an IMU sample");
            }
        }

        let result = {
            let images: Vec<ImageRef> = self
                .cameras
                .iter()
                .enumerate()
                .map(|(index, camera)| {
                    let image = camera.image.as_ref().expect("checked above");
                    ImageRef {
                        pixels: &image.data,
                        width: image.width,
                        height: image.height,
                        encoding: image_encoding(&image.encoding),
                        data_type: ffi::CUV_DATA_UINT8,
                        timestamp_ns: newest,
                        camera_index: index as u32,
                    }
                })
                .collect();
            let depths: Vec<ImageRef> = depth_choice
                .iter()
                .map(|choice| {
                    let depth = match choice {
                        DepthChoice::Passthrough => self.depth.as_ref().expect("checked above"),
                        DepthChoice::Aligned => &self.aligned_depth,
                    };
                    ImageRef {
                        pixels: &depth.data,
                        width: depth.width,
                        height: depth.height,
                        encoding: ffi::CUV_ENCODING_MONO,
                        data_type: ffi::CUV_DATA_UINT16,
                        timestamp_ns: newest,
                        camera_index: images[0].camera_index,
                    }
                })
                .collect();
            vslam.track(&images, &depths)
        };
        self.frames += 1;
        self.last_ts_ns = Some(newest);
        self.clear_frame_set();

        let estimate = match result {
            Ok(estimate) => estimate,
            Err(message) => {
                // The shim catches Track()'s C++ exception; the frame is skipped.
                error!(error = %message, "cuvslam Track failed");
                return None;
            }
        };

        let Some((pose, rig_covariance)) = estimate.world_from_rig else {
            if self.was_tracking {
                self.segment_id += 1;
                self.was_tracking = false;
                // The next tracked frame restarts across an unmeasured gap; its speed
                // against the pre-loss pose would be meaningless.
                self.previous_raw = None;
                warn!(segment = self.segment_id, "cuvslam tracking lost");
            }
            return None;
        };
        // cuVSLAM tracks rig_frame(); the contract is base_frame starting at identity. Both
        // collapse to the raw pose when the two frames are the same.
        let base_from_rig = self.base_from_rig.expect("set with the tracker");
        let rig_from_base = self.rig_from_base.expect("set with the tracker");
        let raw_pose = base_from_rig * cuv_pose_to_isometry(&pose) * rig_from_base;
        // cuVSLAM reports the 6x6 on the rig frame; the published pose is on base_frame,
        // so the covariance moves through the same fixed transform. NaN (the tracker's
        // unconstrained marker) survives the product and still trips the gate below.
        self.covariance = self.covariance_adjoint
            * Matrix6::from_row_slice(&rig_covariance)
            * self.covariance_adjoint.transpose();
        let translation_variances = [
            self.covariance[(0, 0)],
            self.covariance[(1, 1)],
            self.covariance[(2, 2)],
        ];
        let translation_std = translation_variances
            .iter()
            .fold(f64::NEG_INFINITY, |a, b| a.max(*b))
            .sqrt();
        // f64::max skips NaN, so unconstrained frames need the explicit finiteness check.
        let translation_finite = translation_variances.iter().all(|variance| variance.is_finite());
        let mut gate_frame = false;
        if self.config.covariance_gate_translation_std > 0.0
            && self.was_tracking
            && (!translation_finite || translation_std > self.config.covariance_gate_translation_std)
        {
            gate_frame = true;
            self.covariance_gated += 1;
            warn_throttled!(
                Duration::from_secs(5),
                translation_std,
                gated = self.covariance_gated,
                "cuvslam covariance gate holding pose",
            );
        }
        // Speed is judged against the previous raw pose even when that frame was gated:
        // after a teleport the tracker keeps integrating from the far side, so later
        // frames are near the teleported pose and only the jump frame itself trips.
        if self.config.speed_gate_max_linear > 0.0 || self.config.speed_gate_max_angular > 0.0 {
            if let Some(previous_raw) = &self.previous_raw {
                // Track() rejects non-increasing stamps, so dt is strictly positive here.
                let dt = (estimate.timestamp_ns - self.previous_raw_ns) as f64 / 1.0e9;
                let linear_speed = (raw_pose.translation.vector - previous_raw.translation.vector).norm() / dt;
                let angular_speed = previous_raw.rotation.angle_to(&raw_pose.rotation) / dt;
                if (self.config.speed_gate_max_linear > 0.0 && linear_speed > self.config.speed_gate_max_linear)
                    || (self.config.speed_gate_max_angular > 0.0
                        && angular_speed > self.config.speed_gate_max_angular)
                {
                    gate_frame = true;
                    self.speed_gated += 1;
                    warn_throttled!(
                        Duration::from_secs(5),
                        linear_mps = linear_speed,
                        angular_rps = angular_speed,
                        gated = self.speed_gated,
                        "cuvslam speed gate holding pose",
                    );
                }
            }
        }
        self.previous_raw = Some(raw_pose);
        self.previous_raw_ns = estimate.timestamp_ns;
        let rebase = self.rebase.unwrap_or_else(Isometry3::identity);
        let world_from_base = self.world_from_base.unwrap_or_else(Isometry3::identity);
        if gate_frame {
            // Blank wall, repeated texture, teleport: hold the pose and rebase so recovery
            // keeps only post-recovery deltas.
            self.rebase = Some(world_from_base * raw_pose.inverse());
        } else {
            self.world_from_base = Some(rebase * raw_pose);
        }
        self.was_tracking = true;
        self.tracked += 1;
        Some(self.output(estimate.timestamp_ns))
    }

    fn output(&self, timestamp_ns: i64) -> Odometry {
        let world_from_base = self.world_from_base.unwrap_or_else(Isometry3::identity);
        let mut msg = Odometry::default();
        msg.header.stamp = to_stamp(timestamp_ns);
        msg.header.frame_id = self.config.odom_frame.clone();
        msg.child_frame_id = self.config.base_frame.clone();
        let translation: Vector3<f64> = world_from_base.translation.vector;
        msg.pose.pose.position.x = translation.x;
        msg.pose.pose.position.y = translation.y;
        msg.pose.pose.position.z = translation.z;
        let rotation = world_from_base.rotation.quaternion();
        msg.pose.pose.orientation.x = rotation.i;
        msg.pose.pose.orientation.y = rotation.j;
        msg.pose.pose.orientation.z = rotation.k;
        msg.pose.pose.orientation.w = rotation.w;
        // cuVSLAM's own 6x6, row-major xyz-rpy, the order ROS uses, already moved onto
        // base_frame with the pose.
        for row in 0..6 {
            for column in 0..6 {
                msg.pose.covariance[row * 6 + column] = self.covariance[(row, column)];
            }
        }
        msg
    }

    pub fn report(&self) {
        info!(
            frames = self.frames,
            tracked = self.tracked,
            resets = self.segment_id,
            imu_samples = self.imu_samples,
            imu_dropped_before_start = self.imu_dropped,
            skew_rejects = self.skew_rejects,
            depth_reprojected = self.depth_reprojected,
            covariance_gated = self.covariance_gated,
            speed_gated = self.speed_gated,
            unmatched_images = self.unplaced_images,
            "cuvslam shutting down"
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use dimos_module::nalgebra::{Translation3, UnitQuaternion};

    #[test]
    fn adjoint_of_identity_is_identity() {
        assert_eq!(se3_adjoint(&Isometry3::identity()), Matrix6::identity());
    }

    #[test]
    fn adjoint_pure_rotation_has_no_coupling() {
        let rotation = UnitQuaternion::from_euler_angles(0.0, 0.0, std::f64::consts::FRAC_PI_2);
        let adjoint = se3_adjoint(&Isometry3::from_parts(Translation3::identity(), rotation));
        let rotation_matrix = rotation.to_rotation_matrix().into_inner();
        assert!((adjoint.fixed_view::<3, 3>(0, 0) - rotation_matrix).norm() < 1.0e-12);
        assert!(adjoint.fixed_view::<3, 3>(0, 3).norm() < 1.0e-12);
        assert!((adjoint.fixed_view::<3, 3>(3, 3) - rotation_matrix).norm() < 1.0e-12);
        assert!(adjoint.fixed_view::<3, 3>(3, 0).norm() < 1.0e-12);
    }

    #[test]
    fn adjoint_translation_couples_rotation_into_translation() {
        // A yaw-rate uncertainty at a 2 m lever arm reads as lateral translation
        // uncertainty: the (x, yaw) coupling term is skew(t) * R with R = I.
        let lever_arm = Isometry3::from_parts(Translation3::new(2.0, 0.0, 0.0), UnitQuaternion::identity());
        let adjoint = se3_adjoint(&lever_arm);
        assert!((adjoint[(1, 5)] + 2.0).abs() < 1.0e-12); // y <- yaw
        assert!((adjoint[(2, 4)] - 2.0).abs() < 1.0e-12); // z <- pitch
        assert_eq!(adjoint.fixed_view::<3, 3>(0, 0), Matrix3::identity());
    }

    #[test]
    fn adjoint_moves_covariance_between_frames() {
        // Pure yaw variance on the rig maps into y-translation variance on a base
        // displaced 2 m along x: sigma_y = arm^2 * sigma_yaw.
        let lever_arm = Isometry3::from_parts(Translation3::new(2.0, 0.0, 0.0), UnitQuaternion::identity());
        let adjoint = se3_adjoint(&lever_arm);
        let mut rig_covariance = Matrix6::zeros();
        rig_covariance[(5, 5)] = 0.01; // yaw
        let base_covariance = adjoint * rig_covariance * adjoint.transpose();
        assert!((base_covariance[(1, 1)] - 0.04).abs() < 1.0e-12);
        assert!((base_covariance[(5, 5)] - 0.01).abs() < 1.0e-12);
    }

    #[test]
    fn left_swap_when_partner_sits_at_negative_x() {
        let left = Isometry3::from_parts(Translation3::new(0.0, 0.0, 0.0), UnitQuaternion::identity());
        let right = Isometry3::from_parts(Translation3::new(0.12, 0.0, 0.0), UnitQuaternion::identity());
        assert!(!needs_left_swap(&left, &right));
        assert!(needs_left_swap(&right, &left));
    }

    #[test]
    fn odometry_mode_selection() {
        assert_eq!(odometry_mode(Mode::Rgbd, true), ffi::CUV_ODOMETRY_RGBD);
        assert_eq!(odometry_mode(Mode::Mono, false), ffi::CUV_ODOMETRY_MONO);
        assert_eq!(odometry_mode(Mode::Stereo, false), ffi::CUV_ODOMETRY_MULTICAMERA);
        assert_eq!(odometry_mode(Mode::Stereo, true), ffi::CUV_ODOMETRY_INERTIAL);
    }

    #[test]
    fn encoding_declares_color_as_rgb() {
        assert_eq!(image_encoding("mono8"), ffi::CUV_ENCODING_MONO);
        assert_eq!(image_encoding("rgb8"), ffi::CUV_ENCODING_RGB);
        assert_eq!(image_encoding("bgr8"), ffi::CUV_ENCODING_RGB);
    }
}
