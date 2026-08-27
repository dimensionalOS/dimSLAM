// Copyright 2026 Dimensional Inc.
// SPDX-License-Identifier: Apache-2.0
//
// cuVSLAM keeps one world frame for the tracker's life; nothing is emitted while lost.

mod depth_cloud;
mod depth_reproject;
mod msg_convert;

use std::collections::{HashMap, VecDeque};
use std::str::FromStr;
use std::time::Duration;

use nalgebra::{Isometry3, Matrix3, Matrix6};
use serde::{Deserialize, Serialize};
use tracing::{error, info, warn};

use self::depth_cloud::depth_cloud;
use self::depth_reproject::reproject_depth;
use self::msg_convert::{cuv_pose_to_isometry, to_cuv_pose, to_distortion};
use crate::types::{
    CameraModel, ImageFrame, ImuNoiseModel, ImuSample, OdometryEstimate, PointCloud, TfLookup,
};
use crate::{error_throttled, warn_throttled};
use cu_vslam_rs::{ffi, CameraParams, ImageRef, Tracker};

/// cuVSLAM's Track() contract asks for stereo stamps within 1 ms.
const MAX_PAIR_SKEW_NS: i64 = 1_000_000;

/// A stall long enough to overflow this has outlived any use the buffered IMU had.
const MAX_PENDING_IMU: usize = 2048;

#[derive(Clone, Copy, PartialEq)]
enum Mode {
    Stereo,
    Mono,
    Rgbd,
}

impl FromStr for Mode {
    type Err = String;

    fn from_str(name: &str) -> Result<Self, Self::Err> {
        match name {
            "stereo" => Ok(Mode::Stereo),
            "mono" => Ok(Mode::Mono),
            "rgbd" => Ok(Mode::Rgbd),
            other => Err(format!(
                "config: camera_mode must be 'stereo', 'mono' or 'rgbd', got '{other}'"
            )),
        }
    }
}

/// Exact for rotation vectors; ROS's fixed-axis rpy block matches only to first order.
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

/// cuVSLAM wants camera 0 to be the left one: its partner sits at +x (optical x is right).
fn needs_left_swap(
    first_rig_from_camera: &Isometry3<f64>,
    second_rig_from_camera: &Isometry3<f64>,
) -> bool {
    (first_rig_from_camera.inverse() * second_rig_from_camera)
        .translation
        .x
        < 0.0
}

fn odometry_mode(mode: Mode, enable_imu: bool) -> u8 {
    match mode {
        Mode::Rgbd => ffi::CUV_ODOMETRY_RGBD,
        Mode::Mono => ffi::CUV_ODOMETRY_MONO,
        // There is no inertial mono or rgbd.
        Mode::Stereo => {
            if enable_imu {
                ffi::CUV_ODOMETRY_INERTIAL
            } else {
                ffi::CUV_ODOMETRY_MULTICAMERA
            }
        }
    }
}

/// Fed as MONO, a three-channel image makes cuVSLAM read a third of each row and track nothing.
fn image_encoding(encoding: &str) -> u8 {
    if encoding == "mono8" {
        ffi::CUV_ENCODING_MONO
    } else {
        ffi::CUV_ENCODING_RGB
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(default)]
pub struct CuvslamOdometryConfig {
    /// "stereo", "mono" or "rgbd". Mono is accurate only up to scale.
    pub camera_mode: String,
    /// One tf frame per camera in cuVSLAM's index order; empty discovers them off camera_info.
    pub camera_frames: Vec<String>,
    pub rectified: bool,
    /// Needs a libcuvslam built with ENFORCE_GPU=OFF; stock SDK binaries are GPU-only.
    pub use_gpu: bool,
    /// Frame stamped on the emitted odometry; the tracker's world, drifting freely.
    pub odom_frame_id: String,
    pub output_frame_id: String,
    /// Empty means output_frame_id; NVIDIA's examples use the left camera's optical frame.
    pub rig_frame_id: String,
    /// Parent of the loop-closure correction. Nothing here emits map->odom yet, so it is
    /// carried only so a caller's frame naming survives the trip.
    pub map_frame_id: String,
    /// Meters; over this the frame's motion is dropped and later frames rebase onto the held pose. 0 disables.
    pub covariance_gate_translation_std: f64,
    /// A teleport the covariance gate believes: m/s and rad/s against the previous raw pose.
    pub speed_gate_max_linear: f64,
    pub speed_gate_max_angular: f64,
    /// cuVSLAM's Inertial mode: stereo plus one IMU, whose noise model `handle_imu_info` supplies.
    pub enable_imu: bool,
    /// Raw depth units per metre, keyed by the depth image's frame_id. 1000 for sixteen-bit
    /// millimetres. Depth from a frame with no entry is dropped rather than scaled by a
    /// number belonging to another camera.
    pub frame_id_to_depth_units_per_meter: HashMap<String, f64>,
    /// Range gate on the returned depth cloud, metres; 0 leaves it open.
    pub depth_cloud_min_range: f64,
    pub depth_cloud_max_range: f64,
    /// One median point per k x k depth block; <= 1 is off.
    pub depth_cloud_decimation: i64,
}

/// A zeroed default would divide depth by nothing and silently pick mono.
impl Default for CuvslamOdometryConfig {
    fn default() -> Self {
        Self {
            camera_mode: "stereo".to_string(),
            camera_frames: Vec::new(),
            rectified: true,
            use_gpu: true,
            odom_frame_id: "odom".to_string(),
            output_frame_id: "base_link".to_string(),
            rig_frame_id: String::new(),
            map_frame_id: "map".to_string(),
            covariance_gate_translation_std: 0.0,
            speed_gate_max_linear: 0.0,
            speed_gate_max_angular: 0.0,
            enable_imu: false,
            frame_id_to_depth_units_per_meter: HashMap::new(),
            depth_cloud_min_range: 0.0,
            depth_cloud_max_range: 0.0,
            depth_cloud_decimation: 0,
        }
    }
}

struct RigCamera {
    frame: String,
    rig_from_camera: Isometry3<f64>,
    info: CameraModel,
    image: Option<ImageFrame>,
}

enum DepthChoice {
    Passthrough,
    Aligned,
}

pub struct CuvslamCore {
    config: CuvslamOdometryConfig,
    mode: Mode,
    /// The rig, in cuVSLAM's camera order.
    cameras: Vec<RigCamera>,
    camera_info_by_frame: HashMap<String, CameraModel>,
    imu_model: Option<ImuNoiseModel>,
    /// cuVSLAM needs non-decreasing stamps, but a caller's images can overtake its IMU.
    pending_imu: VecDeque<ffi::CuvImuMeasurement>,
    vslam: Option<Tracker>,
    tracker_unbuildable: bool,

    depth: Option<ImageFrame>,
    depth_info: Option<CameraModel>,
    aligned_depth: ImageFrame,
    camera_from_depth: Option<Isometry3<f64>>,
    last_ts_ns: Option<i64>,

    world_from_base: Option<Isometry3<f64>>,
    base_from_rig: Option<Isometry3<f64>>,
    rig_from_base: Option<Isometry3<f64>>,
    /// Adjoint of base_from_rig, fixed with it at tracker creation.
    covariance_adjoint: Matrix6<f64>,
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

impl CuvslamCore {
    /// Rejects an unusable config here rather than at the first image, where a bad
    /// `camera_mode` used to become mono and quietly halve the accuracy.
    pub fn new(config: CuvslamOdometryConfig) -> Result<Self, String> {
        let mode = Mode::from_str(&config.camera_mode)?;
        if config.enable_imu && mode != Mode::Stereo {
            return Err(format!(
                "config: enable_imu requires camera_mode 'stereo'; cuVSLAM has no inertial \
                 {} mode",
                config.camera_mode
            ));
        }
        // cuVSLAM carries one global depth scale factor, so one depth stream is all this
        // mode can serve; a second entry would silently be applied to the wrong camera.
        if mode == Mode::Rgbd && config.frame_id_to_depth_units_per_meter.len() != 1 {
            return Err(format!(
                "config: camera_mode 'rgbd' needs exactly one \
                 frame_id_to_depth_units_per_meter entry, got {}",
                config.frame_id_to_depth_units_per_meter.len()
            ));
        }
        Ok(Self {
            config,
            mode,
            cameras: Vec::new(),
            camera_info_by_frame: HashMap::new(),
            imu_model: None,
            pending_imu: VecDeque::new(),
            vslam: None,
            tracker_unbuildable: false,
            depth: None,
            depth_info: None,
            aligned_depth: ImageFrame::default(),
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
        })
    }

    fn rig_frame(&self) -> &str {
        if self.config.rig_frame_id.is_empty() {
            &self.config.output_frame_id
        } else {
            &self.config.rig_frame_id
        }
    }

    pub fn handle_camera_info(&mut self, info: CameraModel, tf: &dyn TfLookup) {
        if self.vslam.is_some() {
            return; // rig is fixed once the tracker exists
        }
        self.camera_info_by_frame
            .insert(info.frame_id.clone(), info);
        self.resolve_rig(tf);
    }

    /// All-or-nothing: no camera is placed until every camera resolves.
    fn resolve_rig(&mut self, tf: &dyn TfLookup) {
        if !self.cameras.is_empty() {
            return;
        }
        let discovered = self.config.camera_frames.is_empty();
        let mut frames = self.config.camera_frames.clone();
        if discovered {
            frames.extend(self.camera_info_by_frame.keys().cloned());
            // camera_info has no order of its own, and the rig is indexed.
            frames.sort();
            let expected = if self.mode == Mode::Stereo { 2 } else { 1 };
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
            let Some(rig_from_camera) = tf.latest(self.rig_frame(), frame) else {
                return;
            };
            let Some(info) = self.camera_info_by_frame.get(frame) else {
                return;
            };
            cameras.push(RigCamera {
                frame: frame.clone(),
                rig_from_camera,
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
        self.cameras
            .iter()
            .position(|camera| camera.frame == frame_id)
    }

    pub fn handle_imu_info(&mut self, info: ImuNoiseModel, tf: &dyn TfLookup) {
        if self.vslam.is_some() {
            return; // rig is fixed once the tracker exists
        }
        self.imu_model = Some(info);
        self.resolve_rig(tf);
    }

    pub fn handle_imu(&mut self, msg: &ImuSample) {
        if self.vslam.is_none() {
            self.imu_dropped += 1;
            return;
        }
        self.pending_imu.push_back(ffi::CuvImuMeasurement {
            timestamp_ns: msg.timestamp_ns,
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

    pub fn handle_image(&mut self, img: ImageFrame, tf: &dyn TfLookup) -> Option<OdometryEstimate> {
        if self.cameras.is_empty() {
            self.resolve_rig(tf);
        }
        let Some(index) = self.camera_index(&img.frame_id) else {
            self.unplaced_images += 1;
            warn_throttled!(
                Duration::from_secs(10),
                frame_id = %img.frame_id,
                dropped = self.unplaced_images,
                rig_cameras = self.cameras.len(),
                "cuvslam dropping image with a frame_id not on the rig",
            );
            return None;
        };
        self.cameras[index].image = Some(img);
        self.try_track(tf)
    }

    pub fn handle_depth_image(
        &mut self,
        img: ImageFrame,
        tf: &dyn TfLookup,
    ) -> (Option<PointCloud>, Option<OdometryEstimate>) {
        // Downstream indexes by step and height, so an undersized buffer would panic.
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

    /// Scaling depth by a number belonging to another camera turns a metric map into a
    /// plausible-looking wrong one, so an unlisted frame is dropped instead.
    fn depth_units(&self, frame_id: &str) -> Option<f64> {
        let units = self.config.frame_id_to_depth_units_per_meter.get(frame_id);
        if units.is_none() {
            error_throttled!(
                Duration::from_secs(10),
                depth_frame = %frame_id,
                "config: frame_id_to_depth_units_per_meter has no entry for this depth frame",
            );
        }
        units.copied()
    }

    /// A driver's own cloud carries every far, noisy pixel; this one is range-gated.
    fn depth_cloud_msg(&self, depth: &ImageFrame) -> Option<PointCloud> {
        let info = self.depth_info.as_ref().or_else(|| {
            self.cameras
                .iter()
                .find(|camera| camera.frame == depth.frame_id)
                .map(|camera| &camera.info)
        })?;
        let mut cloud = PointCloud::default();
        depth_cloud(
            depth,
            info,
            self.depth_units(&depth.frame_id)?,
            self.config.depth_cloud_min_range,
            self.config.depth_cloud_max_range,
            self.config.depth_cloud_decimation.max(0) as u32,
            &mut cloud,
        );
        Some(cloud)
    }

    pub fn handle_depth_camera_info(&mut self, info: CameraModel) {
        if self.depth_info.is_none() {
            self.depth_info = Some(info);
        }
    }

    /// cuVSLAM's RGBD contract needs depth pixel-aligned with the rig camera.
    fn align_depth(&mut self, tf: &dyn TfLookup) -> Option<DepthChoice> {
        let depth = self.depth.as_ref().expect("checked by try_track");
        let camera = &self.cameras[0];
        if depth.frame_id == camera.frame {
            return Some(DepthChoice::Passthrough);
        }
        let Some(depth_info) = self.depth_info.as_ref() else {
            warn_throttled!(
                Duration::from_secs(10),
                depth_frame = %depth.frame_id,
                camera_frame = %camera.frame,
                "cuvslam: depth is in another frame than the camera and needs \
                 depth_camera_info to reproject",
            );
            return None;
        };
        if self.camera_from_depth.is_none() {
            let Some(camera_from_depth) = tf.latest(&camera.frame, &depth.frame_id) else {
                warn_throttled!(
                    Duration::from_secs(10),
                    depth_frame = %depth.frame_id,
                    camera_frame = %camera.frame,
                    "cuvslam: tf does not connect the depth frame to the camera",
                );
                return None;
            };
            self.camera_from_depth = Some(camera_from_depth);
            info!(
                depth_frame = %depth.frame_id,
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
            self.depth_units(&depth.frame_id)?,
            &mut aligned,
        );
        self.aligned_depth = aligned;
        self.depth_reprojected += 1;
        Some(DepthChoice::Aligned)
    }

    fn ensure_tracker(&mut self, tf: &dyn TfLookup) {
        if self.vslam.is_some() || self.tracker_unbuildable {
            return;
        }
        let Some(base_from_rig) = tf.latest(&self.config.output_frame_id, self.rig_frame()) else {
            warn_throttled!(
                Duration::from_secs(10),
                rig_frame = self.rig_frame(),
                base_frame = %self.config.output_frame_id,
                "cuvslam: tf does not place the rig frame against base_frame",
            );
            return;
        };
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
                    principal: [info.intrinsics[2] as f32, info.intrinsics[5] as f32],
                    focal: [info.intrinsics[0] as f32, info.intrinsics[4] as f32],
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
            let imu_frame = &imu_model.frame_id;
            let Some(rig_from_imu) = tf.latest(self.rig_frame(), imu_frame) else {
                warn_throttled!(
                    Duration::from_secs(10),
                    imu_frame = %imu_frame,
                    "cuvslam: enable_imu is on but tf does not place the IMU",
                );
                return;
            };
            imu_calibration = Some(ffi::CuvImuCalibration {
                rig_from_imu: to_cuv_pose(&rig_from_imu),
                gyroscope_noise_density: imu_model.gyro_noise_density as f32,
                gyroscope_random_walk: imu_model.gyro_random_walk as f32,
                accelerometer_noise_density: imu_model.accel_noise_density as f32,
                accelerometer_random_walk: imu_model.accel_random_walk as f32,
                frequency: imu_model.frequency as f32,
            });
        }

        let mode = self.mode;
        let mut tracker_config = ffi::CuvConfig {
            odometry_mode: odometry_mode(mode, self.config.enable_imu),
            use_gpu: self.config.use_gpu,
            // cuVSLAM: "Rectified stereo camera mode only works with 1+ stereo cameras".
            rectified_stereo_camera: self.config.rectified && mode == Mode::Stereo,
            rgbd_depth_scale_factor: 1.0,
            rgbd_depth_camera_id: -1,
        };
        if mode == Mode::Rgbd {
            let (_, units) = self
                .config
                .frame_id_to_depth_units_per_meter
                .iter()
                .next()
                .expect("rgbd checked for exactly one entry in new()");
            tracker_config.rgbd_depth_scale_factor = *units as f32;
            // align_depth delivers in cameras[0]'s frame; the -1 default silently ignores depth.
            tracker_config.rgbd_depth_camera_id = 0;
        }

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
                        // Nothing about the rig changes later, so a retry would only repeat this.
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

    fn try_track(&mut self, tf: &dyn TfLookup) -> Option<OdometryEstimate> {
        let mode = self.mode;
        if self.cameras.is_empty() || (mode == Mode::Rgbd && self.depth.is_none()) {
            return None;
        }
        if self.cameras.iter().any(|camera| camera.image.is_none()) {
            return None;
        }
        self.ensure_tracker(tf);
        self.vslam.as_ref()?;

        let stamps: Vec<i64> = self
            .cameras
            .iter()
            .map(|camera| camera.image.as_ref().expect("checked above").timestamp_ns)
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
        // A late sample can leave the deque unsorted, so drain from the front.
        while self
            .pending_imu
            .front()
            .is_some_and(|m| m.timestamp_ns <= newest)
        {
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
                // The shim catches Track()'s C++ exception.
                error!(error = %message, "cuvslam Track failed");
                return None;
            }
        };

        let Some((pose, rig_covariance)) = estimate.world_from_rig else {
            if self.was_tracking {
                self.segment_id += 1;
                self.was_tracking = false;
                // Speed against the pre-loss pose would be meaningless across the untracked gap.
                self.previous_raw = None;
                warn!(segment = self.segment_id, "cuvslam tracking lost");
            }
            return None;
        };
        // cuVSLAM tracks rig_frame(); the published pose is base_frame.
        let base_from_rig = self.base_from_rig.expect("set with the tracker");
        let rig_from_base = self.rig_from_base.expect("set with the tracker");
        let raw_pose = base_from_rig * cuv_pose_to_isometry(&pose) * rig_from_base;
        // NaN, the tracker's unconstrained marker, survives the product and still trips the gate.
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
        let translation_finite = translation_variances
            .iter()
            .all(|variance| variance.is_finite());
        let mut gate_frame = false;
        if self.config.covariance_gate_translation_std > 0.0
            && self.was_tracking
            && (!translation_finite
                || translation_std > self.config.covariance_gate_translation_std)
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
        // previous_raw advances even on gated frames: after a teleport only the jump frame trips.
        if self.config.speed_gate_max_linear > 0.0 || self.config.speed_gate_max_angular > 0.0 {
            if let Some(previous_raw) = &self.previous_raw {
                // Track() rejects non-increasing stamps, so dt is strictly positive here.
                let dt = (estimate.timestamp_ns - self.previous_raw_ns) as f64 / 1.0e9;
                let linear_speed =
                    (raw_pose.translation.vector - previous_raw.translation.vector).norm() / dt;
                let angular_speed = previous_raw.rotation.angle_to(&raw_pose.rotation) / dt;
                if (self.config.speed_gate_max_linear > 0.0
                    && linear_speed > self.config.speed_gate_max_linear)
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
            // Hold the pose and rebase so only post-recovery deltas reach the output.
            self.rebase = Some(world_from_base * raw_pose.inverse());
        } else {
            self.world_from_base = Some(rebase * raw_pose);
        }
        self.was_tracking = true;
        self.tracked += 1;
        Some(self.output(estimate.timestamp_ns))
    }

    fn output(&self, timestamp_ns: i64) -> OdometryEstimate {
        OdometryEstimate {
            timestamp_ns,
            frame_id: self.config.odom_frame_id.clone(),
            child_frame_id: self.config.output_frame_id.clone(),
            pose: self.world_from_base.unwrap_or_else(Isometry3::identity),
            pose_covariance: self.covariance,
            ..Default::default()
        }
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
            "cuvslam odometry counters"
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use nalgebra::{Translation3, UnitQuaternion};

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
        let lever_arm =
            Isometry3::from_parts(Translation3::new(2.0, 0.0, 0.0), UnitQuaternion::identity());
        let adjoint = se3_adjoint(&lever_arm);
        assert!((adjoint[(1, 5)] + 2.0).abs() < 1.0e-12); // y <- yaw
        assert!((adjoint[(2, 4)] - 2.0).abs() < 1.0e-12); // z <- pitch
        assert_eq!(adjoint.fixed_view::<3, 3>(0, 0), Matrix3::identity());
    }

    #[test]
    fn adjoint_moves_covariance_between_frames() {
        // A 2 m lever arm turns rig yaw variance into base y variance: var_y = arm^2 * var_yaw.
        let lever_arm =
            Isometry3::from_parts(Translation3::new(2.0, 0.0, 0.0), UnitQuaternion::identity());
        let adjoint = se3_adjoint(&lever_arm);
        let mut rig_covariance = Matrix6::zeros();
        rig_covariance[(5, 5)] = 0.01; // yaw
        let base_covariance = adjoint * rig_covariance * adjoint.transpose();
        assert!((base_covariance[(1, 1)] - 0.04).abs() < 1.0e-12);
        assert!((base_covariance[(5, 5)] - 0.01).abs() < 1.0e-12);
    }

    #[test]
    fn left_swap_when_partner_sits_at_negative_x() {
        let left =
            Isometry3::from_parts(Translation3::new(0.0, 0.0, 0.0), UnitQuaternion::identity());
        let right = Isometry3::from_parts(
            Translation3::new(0.12, 0.0, 0.0),
            UnitQuaternion::identity(),
        );
        assert!(!needs_left_swap(&left, &right));
        assert!(needs_left_swap(&right, &left));
    }

    #[test]
    fn odometry_mode_selection() {
        assert_eq!(odometry_mode(Mode::Rgbd, false), ffi::CUV_ODOMETRY_RGBD);
        assert_eq!(odometry_mode(Mode::Mono, false), ffi::CUV_ODOMETRY_MONO);
        assert_eq!(
            odometry_mode(Mode::Stereo, false),
            ffi::CUV_ODOMETRY_MULTICAMERA
        );
        assert_eq!(
            odometry_mode(Mode::Stereo, true),
            ffi::CUV_ODOMETRY_INERTIAL
        );
    }

    #[test]
    fn an_unknown_camera_mode_is_refused_rather_than_becoming_mono() {
        let config = CuvslamOdometryConfig {
            camera_mode: "steREO".to_string(),
            ..Default::default()
        };
        let error = CuvslamCore::new(config).err().expect("must not build");
        assert!(error.contains("camera_mode"), "{error}");
    }

    #[test]
    fn enable_imu_outside_stereo_is_refused_rather_than_dropped() {
        for camera_mode in ["mono", "rgbd"] {
            let config = CuvslamOdometryConfig {
                camera_mode: camera_mode.to_string(),
                enable_imu: true,
                frame_id_to_depth_units_per_meter: HashMap::from([("depth".to_string(), 1000.0)]),
                ..Default::default()
            };
            let error = CuvslamCore::new(config).err().expect("must not build");
            assert!(error.contains("enable_imu"), "{camera_mode}: {error}");
        }
    }

    #[test]
    fn rgbd_needs_exactly_one_depth_scale() {
        let with_entries = |entries: Vec<(String, f64)>| CuvslamOdometryConfig {
            camera_mode: "rgbd".to_string(),
            frame_id_to_depth_units_per_meter: entries.into_iter().collect(),
            ..Default::default()
        };
        assert!(CuvslamCore::new(with_entries(Vec::new())).is_err());
        assert!(CuvslamCore::new(with_entries(vec![
            ("a".to_string(), 1000.0),
            ("b".to_string(), 1.0),
        ]))
        .is_err());
        assert!(CuvslamCore::new(with_entries(vec![("a".to_string(), 1000.0)])).is_ok());
    }

    #[test]
    fn encoding_declares_color_as_rgb() {
        assert_eq!(image_encoding("mono8"), ffi::CUV_ENCODING_MONO);
        assert_eq!(image_encoding("rgb8"), ffi::CUV_ENCODING_RGB);
        assert_eq!(image_encoding("bgr8"), ffi::CUV_ENCODING_RGB);
    }
}
