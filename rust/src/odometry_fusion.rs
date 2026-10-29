// Copyright 2026 Dimensional Inc.
// SPDX-License-Identifier: Apache-2.0
//
// One error-state Kalman filter fusing odometry sources told apart by their frame_id. With an
// IMU it propagates by inertial integration, without one by a constant-velocity model; every
// source is a measurement of that one state, so the two modes differ only in prediction.

mod eskf;
mod lidar;
pub mod lio_filter;
pub mod plane_fit;
mod point_map;
mod scan_sync;

use std::collections::VecDeque;

use nalgebra::{DVector, Isometry3, Matrix3, Matrix6, Translation3, UnitQuaternion, Vector3};
use serde::{Deserialize, Serialize};
use tracing::{info, warn};

use crate::types::{ImuSample, LidarScan, OdometryEstimate, Twist};
use crate::warn_throttled;
use eskf::{yaw_only, Cov, Filter, ImuBias, ImuInput, Jacobian, State};
use lidar::{CloudSlice, PlaneEvidence, PlaneMap};
use nalgebra::DMatrix;
pub use lidar::{surface_points, voxel_centroids, LidarConfig, LidarMapConfig, Matching, SurfacePoint};
pub use point_map::{to_world, PointMap};
pub use scan_sync::{header_seconds, Level, ScanSync, SyncedScan, STANDARD_GRAVITY};

const NS_PER_SEC: i64 = 1_000_000_000;

/// Raw gyro differences are too noisy for the tangential lever-arm term. A time constant, not
/// a per-sample weight, so the cutoff does not move with IMU rate.
const ANGULAR_ACCEL_TIME_CONSTANT: f64 = 0.025;

/// A non-finite or non-positive variance drops the dimension rather than being divided by.
fn resolve_variance(configured: f64, from_message: f64) -> f64 {
    let variance = if configured < 0.0 {
        from_message
    } else {
        configured
    };
    if variance.is_finite() && variance > 0.0 {
        variance
    } else {
        0.0
    }
}

/// The same pose with z, roll and pitch dropped.
fn planar_pose(pose: &Isometry3<f64>) -> Isometry3<f64> {
    let translation = pose.translation.vector;
    Isometry3::from_parts(
        Translation3::new(translation.x, translation.y, 0.0),
        yaw_only(&pose.rotation),
    )
}

/// What a source's pose means to the filter. Its twist, wherever twist_variances are set,
/// is a body-frame velocity measurement whatever the kind.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SourceKind {
    /// The pose in a world frame of the source's own that does not drift (e.g. lidar-inertial
    /// odometry). Its first message aligns that frame to the fused one; later ones are poses.
    Pose,
    /// The pose is not used, only the twist (e.g. wheel odometry).
    Twist,
    /// A pose that drifts (e.g. a visual tracker): only the motion between consecutive messages
    /// is fused, as a relative measurement against the filter's own pose at the earlier one.
    #[default]
    PoseDelta,
}

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct SourceConfig {
    /// The transform this source's estimates carry. Both halves are needed: two sources can
    /// share a parent. The child is the output frame: twist is read as the body's own.
    pub parent_frame_id: String,
    pub child_frame_id: String,
    pub kind: SourceKind,
    /// Measurement noise, [x y z roll pitch yaw]: <0 message covariance, 0 drop, >0 fixed.
    /// For `pose` the noise of the pose itself, position along the world axes; for `pose_delta`
    /// the noise of one message-to-message motion, in the body frame at the earlier message.
    pub pose_variances: [f64; 6],
    /// [vx vy vz wx wy wz], same convention, body frame.
    pub twist_variances: [f64; 6],
    /// Zero-velocity update: a twist at or under `still_speed` (m/s) and `still_angular_speed`
    /// (rad/s) is fused as zero velocity with `still_variance` in every axis, in place of the
    /// twist itself. It holds only for that message: the motion model keeps its noise, so a
    /// push the wheels cannot see still loosens the pose for the other sources. 0 disables.
    pub still_variance: f64,
    pub still_speed: f64,
    pub still_angular_speed: f64,
}

impl SourceConfig {
    fn uses_pose(&self) -> bool {
        self.kind != SourceKind::Twist && self.pose_variances.iter().any(|v| *v != 0.0)
    }

    fn uses_twist(&self) -> bool {
        self.twist_variances.iter().any(|v| *v != 0.0)
    }
}

/// These all describe the physical IMU rather than the filter itself, so they live together
/// in one object.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct ImuConfig {
    /// The frame_id the IMU's samples carry; samples from any other frame are dropped.
    pub frame_id: String,
    pub gyro_noise_density: f64,
    pub gyro_random_walk: f64,
    pub accel_noise_density: f64,
    pub accel_random_walk: f64,
    /// m/s^2 and rad/s, per axis, that the sensor clips at (a legged robot's footfalls clip
    /// most IMUs' accelerometers); 0 when it never clips. A clipped sample still moves the
    /// filter, but with the excess it could not read as added uncertainty.
    pub accel_saturation: f64,
    pub gyro_saturation: f64,
    /// Samples averaged while stationary to level the filter and take the biases.
    pub init_samples: i64,
    /// rad/s. Bias calibration restarts above this rate, so it belongs above this gyro's own
    /// bias and below any real rotation. A noisier gyro needs it raised or it never inits.
    pub init_gyro_limit: f64,
}

/// The derived default would zero the init figures, which `check_config` rejects. The noise
/// figures stay zero: there is no sane guess, so an unset one has to fail loudly.
impl Default for ImuConfig {
    fn default() -> Self {
        Self {
            frame_id: String::new(),
            gyro_noise_density: 0.0,
            gyro_random_walk: 0.0,
            accel_noise_density: 0.0,
            accel_random_walk: 0.0,
            accel_saturation: 0.0,
            gyro_saturation: 0.0,
            init_samples: 200,
            init_gyro_limit: 0.05,
        }
    }
}

/// How fast the body's velocity can change unannounced, as white-noise acceleration
/// densities: the constant-velocity model's only tuning. Unused with an IMU, whose own noise
/// figures take its place.
#[derive(Clone, Copy, Debug, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct ProcessNoise {
    /// m/s^2/sqrt(Hz).
    pub linear_acceleration: f64,
    /// rad/s^2/sqrt(Hz). With `imu_as_measurement`, how fast the angular velocity state can
    /// change between IMU samples.
    pub angular_acceleration: f64,
    /// m/s^3/sqrt(Hz), `imu_as_measurement` only: how fast the specific force state can change
    /// between IMU samples. A legged body's footfalls want it high.
    pub specific_force_rate: f64,
}

impl Default for ProcessNoise {
    fn default() -> Self {
        Self {
            linear_acceleration: 1.0,
            angular_acceleration: 1.0,
            specific_force_rate: 100.0,
        }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct OdometryFusionConfig {
    pub odom_frame_id: String,
    pub output_frame_id: String,
    /// Output cadence. Each output is the filter predicted to its own stamp, so it is smooth
    /// whatever the sources' rates.
    pub publish_rate: f64,
    /// How far back a late measurement can reach before it is dropped, and how far past the
    /// newest input an output may be predicted.
    pub replay_buffer_seconds: f64,
    /// Outlier gate: reject a measurement whose squared Mahalanobis distance, under the full
    /// innovation covariance, exceeds this per measurement dimension. 0 is no gate.
    pub outlier_rejection_allowed_variance: f64,
    /// Gate on the filter's own state, which can run away with no bad measurement; 0 disables.
    pub max_position_m: f64,
    /// One entry per IMU, told apart by the frame_id its samples carry. Each keeps its own
    /// bias states in the filter, and the newest sample of any IMU drives prediction. Empty
    /// predicts by constant velocity instead (see `process_noise`), seeded level at the
    /// origin from the first source message.
    pub imus: Vec<ImuConfig>,
    /// m/s^2. Seeds `filter.gravity` at init; a future ZUPT is meant to refine it from there.
    pub initial_gravity_estimate: f64,
    pub initial_stds: InitialStds,
    /// The constant-velocity model's noise; only used without IMUs.
    pub process_noise: ProcessNoise,
    /// One entry per external odometry stream, told apart by the transform its estimates carry.
    pub odom_sources: Vec<SourceConfig>,
    /// A ground body: z, roll, pitch and their rates are held at zero in the state itself,
    /// and sources are fused in x, y and yaw only.
    pub planar: bool,
    /// Body [vx vy vz wx wy wz]: >0 is a soft zero-velocity measurement of that axis with this
    /// variance, fused with every source message, e.g. vy for a base that cannot slide sideways;
    /// 0 leaves the axis free. Ungated: it is the model's own assumption, not a sensor. With an
    /// IMU the angular entries are ignored: the gyro measures rate, so they could only drag its bias.
    pub zero_velocity_variances: [f64; 6],
    /// Without an IMU, skip the filter: each source's motion since its previous message moves
    /// the fused pose by its inverse-variance share among the sources heard from in the last
    /// half second, per axis (the pre-0.5 behavior). Twist is not used, so every source needs
    /// pose_variances.
    pub no_imu_blend: bool,
    /// Interacting multiple model: runs a stopped model (pose held, no motion noise) beside
    /// the moving one and outputs their mixture, weighted by how well each explains the data,
    /// so a body at rest stops jittering under its sources' noise while any motion hands back
    /// to the moving model within a few messages. How often the body switches between the two
    /// is learned as it runs: the switches it has made, plus one, over the time it has run.
    pub imm: bool,
    /// One entry per lidar, told apart by the frame_id its scans carry. Each scan's points are
    /// fused directly as point-to-plane measurements against `lidar_map`, which they also build.
    pub lidars: Vec<LidarConfig>,
    pub lidar_map: LidarMapConfig,
    /// With IMUs: angular velocity and specific force become states, each IMU sample a
    /// measurement of them (clipped axes left out), instead of the samples driving the
    /// motion model directly. Robust to a body that shakes or clips its IMU.
    pub imu_as_measurement: bool,
}

/// 1-sigma uncertainty the filter starts with, per state block. The bias std applies to
/// every IMU's gyro and accel bias blocks.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct InitialStds {
    pub position: f64,
    pub velocity: f64,
    pub rotation: f64,
    /// Without an IMU only.
    pub angular_velocity: f64,
    /// m/s^2, `imu_as_measurement` only.
    pub specific_force: f64,
    /// m/s^2, with an IMU: above zero gravity becomes a world-frame vector state starting
    /// this uncertain (0.2 is about a degree of lean), so a slightly mis-mounted IMU leans
    /// gravity instead of sinking or lifting the body. 0 holds gravity straight down.
    pub gravity: f64,
    pub bias: f64,
}

impl Default for InitialStds {
    fn default() -> Self {
        Self {
            position: 0.1,
            velocity: 0.1,
            rotation: 0.05,
            angular_velocity: 0.1,
            specific_force: 0.5,
            gravity: 0.0,
            bias: 0.01,
        }
    }
}

impl OdometryFusionConfig {
    pub fn use_imu(&self) -> bool {
        !self.imus.is_empty()
    }
}

/// The derived default would be all zeros, which `check_config` rejects and the filter needs.
impl Default for OdometryFusionConfig {
    fn default() -> Self {
        Self {
            odom_frame_id: "odom".to_string(),
            output_frame_id: "base_link".to_string(),
            publish_rate: 100.0,
            replay_buffer_seconds: 0.5,
            outlier_rejection_allowed_variance: 25.0,
            max_position_m: 0.0,
            imus: Vec::new(),
            initial_gravity_estimate: 9.8,
            initial_stds: InitialStds::default(),
            process_noise: ProcessNoise::default(),
            odom_sources: Vec::new(),
            planar: false,
            zero_velocity_variances: [0.0; 6],
            no_imu_blend: false,
            imm: false,
            lidars: Vec::new(),
            lidar_map: LidarMapConfig::default(),
            imu_as_measurement: false,
        }
    }
}

#[derive(Clone, Debug)]
struct Measurement {
    source: usize,
    /// Planar-projected in planar mode. For `pose_delta`, marks that this message re-clones.
    pose: Option<Isometry3<f64>>,
    /// `pose_delta`, or any source under `no_imu_blend`: the motion since its previous message.
    delta: Option<Isometry3<f64>>,
    pose_variance: [f64; 6], // 0 means dropped
    twist: Option<Twist>,
    twist_variance: [f64; 6],
}

/// Per-source state that lives in the replay history next to the filter's.
#[derive(Clone, Debug, Default)]
struct SourceRuntime {
    /// `pose`: fused_from_source_world, set from its first message and by a re-align.
    alignment: Option<Isometry3<f64>>,
    /// `pose`: its previous message, to tell a jump in the source from drift in the filter.
    previous: Option<PreviousPose>,
    /// `no_imu_blend`: the fused pose at its previous message, its last stamp and its shares.
    blend_anchor: Option<Isometry3<f64>>,
    blend_last_ns: i64,
    blend_inverse_variance: [f64; 6],
}

#[derive(Clone, Debug)]
struct PreviousPose {
    source_pose: Isometry3<f64>,
    fused_pose: Isometry3<f64>,
    /// The fused [position, rotation] variances then.
    fused_variance: [f64; 6],
    /// Gated as a jump: one more message that moves on from it confirms the jump.
    jumped: bool,
}

/// A source keeps its share of the no-IMU blend until it has been silent this long.
const BLEND_ACTIVITY_TIMEOUT_SECONDS: f64 = 0.5;

// The snapshot alongside the kind dwarfs any variant, so boxing buys nothing.
#[allow(clippy::large_enum_variant)]
#[derive(Clone, Debug)]
enum EventKind {
    Seed,
    Imu(ImuInput),
    Measurement(Measurement),
    Cloud(CloudSlice),
}

/// One filter step and the state it left behind, so a late arrival can replay from its slot.
#[derive(Clone, Debug)]
struct Event {
    ts_ns: i64,
    kind: EventKind,
    state: State,
    covariance: Cov,
    sources: Vec<SourceRuntime>,
    /// `imm`: the moving model, then the stopped one. `state` is their mixture.
    imm: Vec<ImmModel>,
    /// `imm`: how many times the likelier model has changed since init.
    imm_switches: u64,
    /// A cloud slice's points are in the map.
    mapped: bool,
}

#[derive(Clone, Debug)]
struct ImmModel {
    state: State,
    covariance: Cov,
    sources: Vec<SourceRuntime>,
    probability: f64,
}

impl Event {
    fn empty(ts_ns: i64, kind: EventKind) -> Self {
        Self {
            ts_ns,
            kind,
            state: State::default(),
            covariance: Cov::zeros(0, 0),
            sources: Vec::new(),
            imm: Vec::new(),
            imm_switches: 0,
            mapped: false,
        }
    }
}

/// Per-IMU accumulators: stationary init averaging and the lever-arm angular-acceleration
/// estimate, both meaningless across IMUs.
#[derive(Clone, Debug, Default)]
struct ImuRuntime {
    init_gyro_sum: Vector3<f64>,
    init_accel_sum: Vector3<f64>,
    init_samples: u64,
    lever_gyro: Vector3<f64>,
    lever_gyro_ns: i64,
    lever_alpha: Vector3<f64>,
}

/// exp(-d^2 / 2), d the Mahalanobis distance of zero from the model's rates (diagonal): 1 at
/// rest, vanishing for a body moving well past its own uncertainty.
fn zero_velocity_likelihood(model: &ImmModel, layout: eskf::Layout) -> f64 {
    let mut rates: Vec<(usize, f64)> = (0..3).map(|i| (3 + i, model.state.velocity[i])).collect();
    if let Some(col) = layout.angular_velocity_col() {
        rates.extend((0..3).map(|i| (col + i, model.state.angular_velocity[i])));
    }
    let distance_sq: f64 = rates
        .iter()
        .filter(|(col, _)| model.covariance[(*col, *col)] > 0.0)
        .map(|(col, rate)| rate * rate / model.covariance[(*col, *col)])
        .sum();
    (-0.5 * distance_sq).exp()
}

fn likelier(models: &[ImmModel]) -> usize {
    (0..models.len())
        .max_by(|a, b| models[*a].probability.total_cmp(&models[*b].probability))
        .unwrap_or(0)
}

pub struct FusionCore {
    config: OdometryFusionConfig,
    filter: Filter,
    sources: Vec<SourceRuntime>,
    /// Per source, the filter's pose-clone slot: `pose_delta` sources that fuse a pose.
    clone_slots: Vec<Option<usize>>,
    initialized: bool,
    /// When the filter initialized: the IMM's switch rate is counted from here.
    imm_start_ns: i64,
    imu_runtime: Vec<ImuRuntime>,
    events: VecDeque<Event>,
    /// Per-source last raw pose and stamp. A delta belongs to the message stream, so replay
    /// never redoes it.
    previous_source_pose: Vec<Option<(i64, Isometry3<f64>)>>,
    last_publish_ns: i64,
    imu_samples: u64,
    measurements: u64,
    gated: u64,
    /// `pose` messages fused past the gate: the source moved on smoothly, so the filter drifted.
    gate_overrides: u64,
    realigned: u64,
    replayed: u64,
    too_late: u64,
    /// Built from settled slices only, so it lives outside the replay history.
    map: PlaneMap,
    /// Per lidar, the filter's mount-correction slot when its mount is estimated.
    mount_slots: Vec<Option<usize>>,
    /// Slices stamped past the newest IMU sample, waiting for it.
    pending_slices: VecDeque<(i64, CloudSlice)>,
    newest_imu_ns: i64,
    off_clock_scans: u64,
    /// Matched points on near-horizontal surfaces (floors, ground): what holds height.
    level_points: u64,
    slices: u64,
    matched_points: u64,
    gated_points: u64,
}

impl FusionCore {
    pub fn new(config: OdometryFusionConfig) -> Self {
        let imu_runtime = vec![ImuRuntime::default(); config.imus.len()];
        let mut clones = 0;
        let clone_slots = config
            .odom_sources
            .iter()
            .map(|source| {
                (source.kind == SourceKind::PoseDelta && source.uses_pose()).then(|| {
                    clones += 1;
                    clones - 1
                })
            })
            .collect();
        let source_count = config.odom_sources.len();
        let config_lidar_map = config.lidar_map.clone();
        let mut mounts = 0;
        let mount_slots = config
            .lidars
            .iter()
            .map(|lidar| {
                (lidar.mount_rotation_std > 0.0).then(|| {
                    mounts += 1;
                    mounts - 1
                })
            })
            .collect();
        Self {
            config,
            filter: Filter::default(),
            sources: vec![SourceRuntime::default(); source_count],
            clone_slots,
            initialized: false,
            imm_start_ns: 0,
            imu_runtime,
            events: VecDeque::new(),
            previous_source_pose: vec![None; source_count],
            last_publish_ns: 0,
            imu_samples: 0,
            measurements: 0,
            gated: 0,
            gate_overrides: 0,
            realigned: 0,
            replayed: 0,
            too_late: 0,
            map: PlaneMap::new(config_lidar_map),
            mount_slots,
            pending_slices: VecDeque::new(),
            newest_imu_ns: 0,
            off_clock_scans: 0,
            level_points: 0,
            slices: 0,
            matched_points: 0,
            gated_points: 0,
        }
    }

    /// The filter's body frame is output_frame_id, so both IMU vectors go through the mount.
    pub fn handle_imu(&mut self, imu: &ImuSample, base_from_imu: &Isometry3<f64>) {
        let Some(index) = self
            .config
            .imus
            .iter()
            .position(|configured| configured.frame_id == imu.frame_id)
        else {
            warn_throttled!(
                std::time::Duration::from_secs(5),
                frame_id = %imu.frame_id,
                "imu from an unconfigured frame_id dropped. Add an `imus` entry with it.",
            );
            return;
        };
        let finite = imu.angular_velocity.iter().all(|v| v.is_finite())
            && imu.linear_acceleration.iter().all(|v| v.is_finite());
        if !finite {
            warn_throttled!(
                std::time::Duration::from_secs(5),
                frame_id = %imu.frame_id,
                "imu sample with a non-finite reading dropped",
            );
            return;
        }
        let lever = base_from_imu.translation.vector;
        let base_from_imu = base_from_imu.rotation;
        let gyro = base_from_imu * imu.angular_velocity;
        let mut accel = base_from_imu * imu.linear_acceleration;
        let ts_ns = imu.timestamp_ns;
        let mut period = 0.0;
        {
            let runtime = &mut self.imu_runtime[index];
            if runtime.lever_gyro_ns != 0 {
                period = (ts_ns - runtime.lever_gyro_ns) as f64 / NS_PER_SEC as f64;
            }
            if runtime.lever_gyro_ns != 0 {
                let dt = (ts_ns - runtime.lever_gyro_ns) as f64 / NS_PER_SEC as f64;
                if dt > 0.0 {
                    let raw_alpha = (gyro - runtime.lever_gyro) / dt;
                    let blend = 1.0 - (-dt / ANGULAR_ACCEL_TIME_CONSTANT).exp();
                    runtime.lever_alpha += (raw_alpha - runtime.lever_alpha) * blend;
                }
            }
            runtime.lever_gyro = gyro;
            runtime.lever_gyro_ns = ts_ns;
            // The lever arm adds centripetal and tangential terms: mount kinematics, not base motion.
            accel -= gyro.cross(&gyro.cross(&lever)) + runtime.lever_alpha.cross(&lever);
        }
        let config = &self.config.imus[index];
        // 1% short of the clip, as a clipped reading jitters just under it.
        let clipped = |reading: &Vector3<f64>, clip: f64| {
            clip > 0.0 && reading.iter().any(|value| value.abs() >= 0.99 * clip)
        };
        let input = ImuInput {
            imu: index,
            gyro,
            accel,
            accel_saturated: clipped(&imu.linear_acceleration, config.accel_saturation),
            gyro_saturated: clipped(&imu.angular_velocity, config.gyro_saturation),
            period,
        };
        if !self.initialized {
            self.check_config();
            if gyro.norm() > self.config.imus[index].init_gyro_limit {
                warn_throttled!(
                    std::time::Duration::from_secs(5),
                    frame_id = %imu.frame_id,
                    gyro_norm = gyro.norm(),
                    init_gyro_limit = self.config.imus[index].init_gyro_limit,
                    "imu init deferred: waiting for the robot to hold still. Hold it \
                     still, or raise init_gyro_limit past the rate above.",
                );
                // The robot moved, which voids every IMU's average, not just this one's.
                for runtime in &mut self.imu_runtime {
                    runtime.init_gyro_sum = Vector3::zeros();
                    runtime.init_accel_sum = Vector3::zeros();
                    runtime.init_samples = 0;
                }
                return;
            }
            {
                let runtime = &mut self.imu_runtime[index];
                runtime.init_gyro_sum += gyro;
                runtime.init_accel_sum += accel;
                runtime.init_samples += 1;
            }
            let all_ready = self
                .imu_runtime
                .iter()
                .zip(&self.config.imus)
                .all(|(runtime, config)| runtime.init_samples >= config.init_samples.max(1) as u64);
            if all_ready {
                let mean_accels: Vec<Vector3<f64>> = self
                    .imu_runtime
                    .iter()
                    .map(|runtime| runtime.init_accel_sum / runtime.init_samples as f64)
                    .collect();
                // Stationary accel reads the specific force R^T (0,0,g): level so it maps to +z.
                // Averaging over IMUs halves what one sensor's bias does to the leveling.
                let combined: Vector3<f64> =
                    mean_accels.iter().sum::<Vector3<f64>>() / mean_accels.len() as f64;
                let mut world_from_body =
                    UnitQuaternion::rotation_between(&combined, &Vector3::z()).unwrap_or_default();
                // A planar body is level by definition, so any tilt the IMU reads is its mount's.
                if self.config.planar {
                    world_from_body = yaw_only(&world_from_body);
                }
                // Any stationary reading beyond g is that IMU's accel bias; left at zero it
                // dead-reckons z away.
                let expected = world_from_body.inverse_transform_vector(&Vector3::new(
                    0.0,
                    0.0,
                    self.config.initial_gravity_estimate,
                ));
                let biases = self
                    .imu_runtime
                    .iter()
                    .zip(&mean_accels)
                    .map(|(runtime, mean_accel)| ImuBias {
                        gyro: runtime.init_gyro_sum / runtime.init_samples as f64,
                        accel: mean_accel - expected,
                    })
                    .collect();
                self.initialize(ts_ns, world_from_body, biases, Some(input));
            }
            return;
        }
        self.insert(Event::empty(ts_ns, EventKind::Imu(input)));
        self.imu_samples += 1;
        self.newest_imu_ns = self.newest_imu_ns.max(ts_ns);
        self.release_slices();
    }

    /// With an IMU, a slice waits until the IMU has reached its stamp, so it lands after the
    /// samples it needs rather than ahead of them, where each arriving sample would replay it.
    fn release_slices(&mut self) {
        let ready = self
            .pending_slices
            .partition_point(|(ts_ns, _)| *ts_ns <= self.newest_imu_ns);
        if ready == 0 {
            return;
        }
        let events = self
            .pending_slices
            .drain(..ready)
            .map(|(ts_ns, slice)| Event::empty(ts_ns, EventKind::Cloud(slice)))
            .collect();
        self.insert_all(events);
    }

    /// `base_from_lidar` carries the scan into the output frame, the filter's body.
    pub fn handle_cloud(&mut self, scan: &LidarScan, base_from_lidar: &Isometry3<f64>) {
        let Some(index) = self
            .config
            .lidars
            .iter()
            .position(|configured| configured.frame_id == scan.frame_id)
        else {
            warn_throttled!(
                std::time::Duration::from_secs(5),
                frame_id = %scan.frame_id,
                "scan from an unconfigured frame_id dropped. Add a `lidars` entry with it.",
            );
            return;
        };
        let slices = lidar::slice_scan(scan, index, &self.config.lidars[index], base_from_lidar);
        let Some((first_ns, _)) = slices.first() else {
            return;
        };
        if !self.initialized {
            if self.config.use_imu() {
                return;
            }
            self.check_config();
            self.initialize(*first_ns, UnitQuaternion::identity(), Vec::new(), None);
        }
        // A driver that stamps a sweep on another clock (one glitching to host time, say) would
        // park its slices past every IMU sample, or, without an IMU, run the filter into the
        // future: past the replay buffer ahead of the newest input, the stamp is not this clock's.
        let newest_ns = self.events.back().map_or(*first_ns, |e| e.ts_ns);
        let ahead = (*first_ns - newest_ns) as f64 / 1.0e9;
        if self.initialized && ahead > self.config.replay_buffer_seconds {
            self.off_clock_scans += 1;
            warn_throttled!(
                std::time::Duration::from_secs(10),
                frame_id = %scan.frame_id,
                ahead_seconds = ahead,
                dropped = self.off_clock_scans,
                "scan stamped ahead of every other input dropped: its driver is on another clock",
            );
            return;
        }
        self.slices += slices.len() as u64;
        if self.config.use_imu() {
            for (ts_ns, slice) in slices {
                let slot = self.pending_slices.partition_point(|(t, _)| *t <= ts_ns);
                self.pending_slices.insert(slot, (ts_ns, slice));
            }
            self.release_slices();
            return;
        }
        self.insert_all(
            slices
                .into_iter()
                .map(|(ts_ns, slice)| Event::empty(ts_ns, EventKind::Cloud(slice)))
                .collect(),
        );
    }

    pub fn handle_source(&mut self, msg: &OdometryEstimate) {
        if !self.initialized {
            if self.config.use_imu() {
                return;
            }
            self.check_config();
            self.initialize(
                msg.timestamp_ns,
                UnitQuaternion::identity(),
                Vec::new(),
                None,
            );
        }
        let Some((source, source_config)) =
            self.config
                .odom_sources
                .iter()
                .enumerate()
                .find(|(_, source_config)| {
                    source_config.parent_frame_id == msg.frame_id
                        && source_config.child_frame_id == msg.child_frame_id
                })
        else {
            warn_throttled!(
                std::time::Duration::from_secs(5),
                frame_id = %msg.frame_id,
                child_frame_id = %msg.child_frame_id,
                "odometry from an unconfigured transform dropped. Add an `odom_sources` entry \
                 with this parent_frame_id and child_frame_id.",
            );
            return;
        };

        let mut pose_variance = [0.0; 6];
        let mut twist_variance = [0.0; 6];
        for dim in 0..6 {
            pose_variance[dim] = resolve_variance(
                source_config.pose_variances[dim],
                msg.pose_covariance[(dim, dim)],
            );
            twist_variance[dim] = resolve_variance(
                source_config.twist_variances[dim],
                msg.twist_covariance[(dim, dim)],
            );
        }

        // A caller building a UnitQuaternion from a zeroed message gets NaN, not an error, and
        // one NaN reaching the state wedges the filter for good: nothing downstream recovers.
        let pose_finite = msg.pose.translation.vector.iter().all(|v| v.is_finite())
            && msg.pose.rotation.coords.iter().all(|v| v.is_finite());
        let twist_finite = msg.twist.linear.iter().all(|v| v.is_finite())
            && msg.twist.angular.iter().all(|v| v.is_finite());
        let pose = (source_config.uses_pose() && pose_finite).then(|| {
            if self.config.planar {
                planar_pose(&msg.pose)
            } else {
                msg.pose
            }
        });
        let mut delta = None;
        let tracks_delta = source_config.kind == SourceKind::PoseDelta || self.blending();
        if tracks_delta {
            if let (Some(pose), Some((previous_ns, previous))) =
                (pose, self.previous_source_pose[source])
            {
                if msg.timestamp_ns <= previous_ns {
                    warn_throttled!(
                        std::time::Duration::from_secs(5),
                        frame_id = %msg.frame_id,
                        "pose_delta message not newer than its predecessor dropped",
                    );
                    return;
                }
                delta = Some(previous.inverse() * pose);
            }
        }

        let twist = (source_config.uses_twist() && twist_finite).then_some(msg.twist);
        let inserted = self.insert(Event::empty(
            msg.timestamp_ns,
            EventKind::Measurement(Measurement {
                source,
                pose,
                delta,
                pose_variance,
                twist,
                twist_variance,
            }),
        ));
        if inserted && tracks_delta {
            if let Some(pose) = pose {
                self.previous_source_pose[source] = Some((msg.timestamp_ns, pose));
            }
        }
        self.measurements += 1;
    }

    fn check_config(&self) {
        for source in &self.config.odom_sources {
            let name = &source.parent_frame_id;
            assert!(
                !source.parent_frame_id.trim().is_empty()
                    && !source.child_frame_id.trim().is_empty(),
                "an `odom_sources` entry needs both parent_frame_id and child_frame_id set"
            );
            assert!(
                source.kind != SourceKind::Twist || source.pose_variances == [0.0; 6],
                "source {name:?} is kind twist, which fuses no pose: zero its pose_variances"
            );
            assert!(
                source.uses_pose() || source.uses_twist(),
                "source {name:?} fuses nothing: set pose_variances or twist_variances"
            );
            assert!(
                !self.config.no_imu_blend || source.uses_pose(),
                "source {name:?} fuses no pose, and no_imu_blend blends poses only"
            );
        }
        for source in &self.config.odom_sources {
            assert!(
                !self.config.imm || source.still_variance == 0.0,
                "source {:?} sets still_variance, but the imm's stopped model already is the \
                 zero-velocity hypothesis: zero one or the other",
                source.parent_frame_id
            );
        }
        assert!(
            !(self.config.no_imu_blend && self.config.imm),
            "no_imu_blend replaces the filter, so it has no models to mix: turn imm off"
        );
        assert!(
            !(self.config.no_imu_blend && self.config.use_imu()),
            "no_imu_blend replaces the filter, so it cannot take imus: empty one or the other"
        );
        if !self.config.use_imu() {
            let noise = &self.config.process_noise;
            assert!(
                noise.linear_acceleration > 0.0 && noise.angular_acceleration > 0.0,
                "without an imu, process_noise needs both densities above zero, or the filter \
                 grows overconfident and gates every source out"
            );
        }
        for (index, lidar) in self.config.lidars.iter().enumerate() {
            let frame_id = &lidar.frame_id;
            assert!(
                !frame_id.trim().is_empty(),
                "every `lidars` entry needs its frame_id set"
            );
            assert!(
                self.config.lidars[..index]
                    .iter()
                    .all(|other| other.frame_id != *frame_id),
                "two `lidars` entries share frame_id {frame_id:?}"
            );
            assert!(
                lidar.point_std > 0.0 && lidar.scan_voxel_size > 0.0 && lidar.slice_seconds > 0.0,
                "lidar {frame_id:?} needs point_std, scan_voxel_size and slice_seconds above zero"
            );
        }
        let map = &self.config.lidar_map;
        assert!(
            map.voxel_size > 0.0 && map.max_plane_std > 0.0 && map.min_points_per_plane >= 3,
            "lidar_map needs voxel_size and max_plane_std above zero and min_points_per_plane >= 3"
        );
        for (index, imu) in self.config.imus.iter().enumerate() {
            let frame_id = &imu.frame_id;
            assert!(
                !frame_id.trim().is_empty(),
                "every `imus` entry needs its frame_id set"
            );
            assert!(
                self.config.imus[..index]
                    .iter()
                    .all(|other| other.frame_id != *frame_id),
                "two `imus` entries share frame_id {frame_id:?}"
            );
            assert!(
                imu.gyro_noise_density > 0.0
                    && imu.gyro_random_walk > 0.0
                    && imu.accel_noise_density > 0.0
                    && imu.accel_random_walk > 0.0,
                "imu {frame_id:?} needs all four noise figures set"
            );
            assert!(
                imu.init_gyro_limit > 0.0,
                "imu {frame_id:?} needs init_gyro_limit above zero or the filter never leaves init"
            );
        }
    }

    fn initialize(
        &mut self,
        timestamp_ns: i64,
        world_from_body: UnitQuaternion<f64>,
        biases: Vec<ImuBias>,
        input: Option<ImuInput>,
    ) {
        self.filter.noise = self
            .config
            .imus
            .iter()
            .map(|imu| eskf::Noise {
                gyro_noise_density: imu.gyro_noise_density,
                gyro_random_walk: imu.gyro_random_walk,
                accel_noise_density: imu.accel_noise_density,
                accel_random_walk: imu.accel_random_walk,
                accel_saturation: imu.accel_saturation,
                gyro_saturation: imu.gyro_saturation,
            })
            .collect();
        self.filter.process_noise = self.config.process_noise;
        self.filter.gravity = self.config.initial_gravity_estimate;
        self.filter.planar = self.config.planar;
        self.filter.layout.clones = self.clone_slots.iter().flatten().count();
        self.filter.layout.imu_states = self.config.imu_as_measurement && !biases.is_empty();
        self.filter.layout.mounts = self.mount_slots.iter().flatten().count();
        self.filter.layout.gravity_state =
            !biases.is_empty() && self.config.initial_stds.gravity > 0.0;
        self.filter.mount_stds = self
            .config
            .lidars
            .iter()
            .filter(|lidar| lidar.mount_rotation_std > 0.0)
            .map(|lidar| lidar.mount_rotation_std)
            .collect();
        let gyro_bias_norms: Vec<f64> = biases.iter().map(|bias| bias.gyro.norm()).collect();
        self.filter.init(
            timestamp_ns,
            world_from_body,
            biases,
            input,
            &self.config.initial_stds,
        );
        self.initialized = true;
        self.imm_start_ns = timestamp_ns;
        self.last_publish_ns = timestamp_ns;
        let mut seed = Event::empty(timestamp_ns, EventKind::Seed);
        self.snapshot(&mut seed);
        if self.imm() {
            let model = ImmModel {
                state: seed.state.clone(),
                covariance: seed.covariance.clone(),
                sources: seed.sources.clone(),
                probability: 0.5,
            };
            seed.imm = vec![model.clone(), model];
        }
        self.events.push_back(seed);
        info!(
            gyro_biases = ?gyro_bias_norms,
            use_imu = self.config.use_imu(),
            planar = self.config.planar,
            "odometry_fusion initialized"
        );
    }

    /// False when the event is older than the replay buffer and was dropped.
    fn insert(&mut self, event: Event) -> bool {
        let newest = self.events.back().expect("seeded at init").ts_ns;
        if event.ts_ns >= newest {
            self.events.push_back(event);
            let index = self.events.len() - 1;
            self.process_at(index);
        } else if event.ts_ns <= self.events.front().expect("seeded at init").ts_ns {
            self.too_late += 1;
            warn_throttled!(
                std::time::Duration::from_secs(10),
                dropped = self.too_late,
                replay_buffer_seconds = self.config.replay_buffer_seconds,
                "measurement older than the replay buffer dropped. Raise \
                 replay_buffer_seconds, or cut the source's latency.",
            );
            return false;
        } else {
            let slot = self.events.partition_point(|e| e.ts_ns <= event.ts_ns);
            self.events.insert(slot, event);
            for index in slot..self.events.len() {
                self.process_at(index);
            }
            self.replayed += 1;
        }
        self.trim_history();
        true
    }

    /// Inserts events stamped in order, replaying once from the earliest rather than once per
    /// event: a scan's slices all land behind the IMU samples that arrived during the sweep.
    fn insert_all(&mut self, events: Vec<Event>) {
        let oldest = self.events.front().expect("seeded at init").ts_ns;
        let mut earliest = None;
        for event in events {
            if event.ts_ns <= oldest {
                self.too_late += 1;
                continue;
            }
            let slot = self.events.partition_point(|e| e.ts_ns <= event.ts_ns);
            self.events.insert(slot, event);
            earliest = Some(earliest.map_or(slot, |e: usize| e.min(slot)));
        }
        let Some(earliest) = earliest else {
            return;
        };
        if earliest + 1 < self.events.len() {
            self.replayed += 1;
        }
        for index in earliest..self.events.len() {
            self.process_at(index);
        }
        // A slice joins the map once, at the pose its first update left: what is ahead of the
        // body is then matchable within the next slice, as it must be going down stairs. A
        // replay from a late source re-fuses the slice but cannot move points already mapped.
        for index in earliest..self.events.len() {
            if self.events[index].mapped {
                continue;
            }
            self.events[index].mapped = true;
            if let EventKind::Cloud(slice) = &self.events[index].kind {
                let (slice, state) = (slice.clone(), self.events[index].state.clone());
                self.map_slice(&slice, &state);
            }
        }
        self.trim_history();
    }

    /// Drops what has left the replay buffer.
    fn trim_history(&mut self) {
        let horizon = self.events.back().expect("nonempty").ts_ns
            - (self.config.replay_buffer_seconds * 1.0e9) as i64;
        while self.events.len() > 1 && self.events.front().expect("nonempty").ts_ns < horizon {
            self.events.pop_front();
        }
    }

    /// A lidar's estimated mount correction in `state`, or none.
    fn mount_of(&self, lidar: usize, state: &State) -> UnitQuaternion<f64> {
        self.mount_slots
            .get(lidar)
            .copied()
            .flatten()
            .and_then(|slot| state.mounts.get(slot).copied())
            .unwrap_or_else(UnitQuaternion::identity)
    }

    fn map_slice(&mut self, slice: &CloudSlice, state: &State) {
        let angular_velocity = self.angular_velocity_of(state);
        let mount = self.mount_of(slice.lidar, state);
        for point in &slice.points {
            let landed = lidar::land(
                point,
                &slice.origin,
                &mount,
                &state.position,
                &state.q,
                &state.velocity,
                &angular_velocity,
            );
            self.map.insert(&landed.world);
        }
        self.map.prune(&state.position);
    }

    /// Body-frame rate of a stored state: its own without an IMU, else its held gyro sample.
    fn angular_velocity_of(&self, state: &State) -> Vector3<f64> {
        if self.filter.layout.angular_velocity_col().is_some() {
            return state.angular_velocity;
        }
        state
            .input
            .as_ref()
            .map_or(Vector3::zeros(), |input| input.gyro - state.biases[input.imu].gyro)
    }

    /// Point-to-plane against the map: an iterated update over the slice's matched points,
    /// each iteration re-matching the points from where the last one left the pose.
    fn apply_cloud(&mut self, slice: &CloudSlice) {
        let prior = self.filter.x.clone();
        let prior_cov = self.filter.p_cov.clone();
        let (angular_velocity, _) = self.filter.angular_velocity();
        let mount_slot = self.mount_slots.get(slice.lidar).copied().flatten();
        let mut cols = vec![0, 1, 2, 6, 7, 8];
        if let Some(slot) = mount_slot {
            let col = self.filter.layout.mount_col(slot);
            cols.extend(col..col + 3);
        }
        let block = DMatrix::from_fn(cols.len(), cols.len(), |r, c| prior_cov[(cols[r], cols[c])]);
        let point_variance = self.config.lidars[slice.lidar].point_std.powi(2);
        let gate = self.map.config().point_gate;
        let iterations = self.map.config().iterations.max(1);
        for iteration in 0..iterations {
            let x = self.filter.x.clone();
            let mount = self.mount_of(slice.lidar, &x);
            let mut evidence = PlaneEvidence::new(cols.len());
            let mut level = 0u64;
            for point in &slice.points {
                let landed = lidar::land(
                    point,
                    &slice.origin,
                    &mount,
                    &x.position,
                    &x.q,
                    &x.velocity,
                    &angular_velocity,
                );
                let Some(plane) = self.map.plane_at(&landed.world) else {
                    continue;
                };
                let residual = -(plane.normal.dot(&landed.world) + plane.offset);
                let row = lidar::plane_row(&plane, &x.q, &landed, mount_slot.is_some());
                let before = evidence.matched;
                evidence.add(
                    &row,
                    residual,
                    point_variance + plane.variance,
                    &block,
                    gate,
                );
                if evidence.matched > before && plane.normal.z.abs() > 0.7 {
                    level += 1;
                }
            }
            let settle = iteration + 1 == iterations;
            if settle {
                self.level_points += level;
                self.matched_points += evidence.matched as u64;
                self.gated_points += evidence.gated as u64;
            }
            if evidence.matched == 0 {
                // Nothing matched from here: keep the prior, untouched.
                self.filter.x = prior;
                self.filter.p_cov = prior_cov;
                return;
            }
            self.filter.update_pose_information(
                &prior,
                &prior_cov,
                &cols,
                &evidence.information,
                &evidence.weighted_residual,
                evidence.weighted_residual_sq,
                evidence.log_det_noise,
                evidence.matched,
                settle,
            );
        }
    }

    fn process_at(&mut self, index: usize) {
        if self.imm() {
            self.process_at_imm(index);
            return;
        }
        {
            let previous = &self.events[index - 1];
            self.filter.x = previous.state.clone();
            self.filter.p_cov = previous.covariance.clone();
            self.sources = previous.sources.clone();
        }
        self.step(index);
        let ts_ns = self.events[index].ts_ns;
        let mut event = std::mem::replace(
            &mut self.events[index],
            Event::empty(ts_ns, EventKind::Seed),
        );
        self.snapshot(&mut event);
        self.events[index] = event;
    }

    fn imm(&self) -> bool {
        self.config.imm
    }

    /// One IMM cycle: mix the models' priors through the switch probabilities, step each under
    /// its own motion model, reweight each by its measurement likelihood, output the mixture.
    fn process_at_imm(&mut self, index: usize) {
        let ts_ns = self.events[index].ts_ns;
        let models = self.events[index - 1].imm.clone();
        let previous = &self.events[index - 1];
        let dt = (ts_ns - previous.ts_ns) as f64 / 1.0e9;
        let switches = previous.imm_switches;
        let previous_likelier = likelier(&models);
        // The switch rate's posterior mean under a flat prior: one switch more than seen, over
        // the time run. Capped at a coin flip, which is no memory at all.
        let elapsed = (ts_ns - self.imm_start_ns) as f64 / 1.0e9;
        let rate = (switches + 1) as f64 / elapsed;
        let switch = (1.0 - (-rate * dt).exp()).min(0.5);
        // Stopping takes being slow first: moving -> stopped is scaled by how likely the moving
        // model finds zero velocity, else a fast body is pulled towards rest every step.
        let stopping = switch * zero_velocity_likelihood(&models[0], self.filter.layout);
        let transition = |from: usize, to: usize| match (from, to) {
            (0, 1) => stopping,
            (0, _) => 1.0 - stopping,
            (_, 0) => switch,
            _ => 1.0 - switch,
        };
        // A body that starts moving starts at rates nothing has measured yet: the stopped
        // model's certain zero would hold the moving one back, so a start carries the moving
        // model's own rate uncertainty, which its process noise kept growing.
        let mut starting = models[1].covariance.clone();
        let mut rate_cols: Vec<usize> = (3..6).collect();
        if let Some(col) = self.filter.layout.angular_velocity_col() {
            rate_cols.extend(col..col + 3);
        }
        for &row in &rate_cols {
            for &col in &rate_cols {
                starting[(row, col)] = models[0].covariance[(row, col)];
            }
        }
        let mut stepped = Vec::new();
        let mut log_weights = Vec::new();
        let mut moving_counters = (0, 0, 0);
        for to in 0..models.len() {
            let predicted: f64 = (0..models.len())
                .map(|from| transition(from, to) * models[from].probability)
                .sum();
            let priors: Vec<(&State, &Cov, f64)> = models
                .iter()
                .enumerate()
                .map(|(from, model)| {
                    let weight = if predicted > 0.0 {
                        transition(from, to) * model.probability / predicted
                    } else {
                        f64::from(u8::from(from == to))
                    };
                    let covariance = if from == 1 && to == 0 {
                        &starting
                    } else {
                        &model.covariance
                    };
                    (&model.state, covariance, weight)
                })
                .collect();
            self.filter.mix(&priors);
            let heaviest = (0..models.len())
                .max_by(|a, b| priors[*a].2.total_cmp(&priors[*b].2))
                .expect("two models");
            self.sources = models[heaviest].sources.clone();
            self.filter.stopped = to == 1;
            self.filter.log_likelihood = 0.0;
            self.step(index);
            log_weights.push(predicted.ln() + self.filter.log_likelihood);
            stepped.push(ImmModel {
                state: self.filter.x.clone(),
                covariance: self.filter.p_cov.clone(),
                sources: self.sources.clone(),
                probability: 0.0,
            });
            if to == 0 {
                moving_counters = (self.gated, self.gate_overrides, self.realigned);
            }
        }
        // The counters report the moving model; the stopped one would double them.
        (self.gated, self.gate_overrides, self.realigned) = moving_counters;
        self.filter.stopped = false;
        // Normalized in log space: a rejected outlier's likelihood underflows f64.
        let finite_max = log_weights
            .iter()
            .cloned()
            .filter(|w| w.is_finite())
            .fold(f64::NEG_INFINITY, f64::max);
        let weights: Vec<f64> = log_weights
            .iter()
            .map(|w| {
                if finite_max.is_finite() && w.is_finite() {
                    (w - finite_max).exp()
                } else {
                    0.0
                }
            })
            .collect();
        let total: f64 = weights.iter().sum();
        for (model, (weight, previous)) in stepped.iter_mut().zip(weights.iter().zip(&models)) {
            // No usable evidence either way: the switch prior alone.
            model.probability = if total > 0.0 {
                weight / total
            } else {
                previous.probability
            };
        }
        let mixture: Vec<(&State, &Cov, f64)> = stepped
            .iter()
            .map(|model| (&model.state, &model.covariance, model.probability))
            .collect();
        self.filter.mix(&mixture);
        let heaviest = (0..stepped.len())
            .max_by(|a, b| stepped[*a].probability.total_cmp(&stepped[*b].probability))
            .expect("two models");
        self.sources = stepped[heaviest].sources.clone();
        let mut event = std::mem::replace(
            &mut self.events[index],
            Event::empty(ts_ns, EventKind::Seed),
        );
        self.snapshot(&mut event);
        event.imm_switches = switches + u64::from(likelier(&stepped) != previous_likelier);
        event.imm = stepped;
        self.events[index] = event;
    }

    fn step(&mut self, index: usize) {
        // Logs once per event, not again each time a late arrival replays it.
        let replaying = index + 1 < self.events.len();
        let ts_ns = self.events[index].ts_ns;
        if !self.filter.predict_to(ts_ns) && !replaying {
            warn_throttled!(
                std::time::Duration::from_secs(10),
                "imu gap over a second left unbridged: the pose held through it. Check the \
                 IMU stream.",
            );
        }
        let kind = self.events[index].kind.clone();
        match kind {
            EventKind::Seed => {}
            EventKind::Imu(input) if self.filter.layout.imu_states => {
                self.filter.update_imu(&input)
            }
            EventKind::Imu(input) => self.filter.x.input = Some(input),
            EventKind::Measurement(measurement) if self.blending() => {
                self.blend(&measurement, ts_ns)
            }
            EventKind::Measurement(measurement) => self.apply(&measurement, replaying),
            EventKind::Cloud(slice) => self.apply_cloud(&slice),
        }
    }

    fn blending(&self) -> bool {
        self.config.no_imu_blend && !self.config.use_imu()
    }

    fn apply(&mut self, measurement: &Measurement, replaying: bool) {
        let source = measurement.source;
        match self.config.odom_sources[source].kind {
            SourceKind::Pose => {
                if let Some(pose) = &measurement.pose {
                    self.fuse_pose(source, pose, &measurement.pose_variance, replaying);
                }
            }
            SourceKind::PoseDelta => {
                if let Some(slot) = self.clone_slots[source] {
                    if let Some(delta) = &measurement.delta {
                        self.fuse_pose_delta(slot, delta, &measurement.pose_variance);
                    }
                    if measurement.pose.is_some() {
                        self.filter.clone_pose(slot);
                    }
                }
            }
            SourceKind::Twist => {}
        }
        if let Some(twist) = &measurement.twist {
            let config = &self.config.odom_sources[source];
            let still = config.still_variance > 0.0
                && twist.linear.norm() <= config.still_speed
                && twist.angular.norm() <= config.still_angular_speed;
            if still {
                let mut variance = [config.still_variance; 6];
                if self.config.use_imu() {
                    // As below: against the gyro, a zero rate only drags its bias.
                    variance[3..].fill(0.0);
                }
                self.fuse_twist(&Twist::default(), &variance);
            } else {
                self.fuse_twist(twist, &measurement.twist_variance);
            }
        }
        let mut constraint = self.config.zero_velocity_variances;
        if self.config.use_imu() {
            // A zero-rate row against the gyro moves only its bias, so real rocking would tilt the filter.
            constraint[3..].fill(0.0);
        }
        if constraint.iter().any(|variance| *variance > 0.0) {
            let rows = self.twist_rows(&Twist::default(), &constraint);
            if !rows.is_empty() {
                let (residual, jacobian, variance) = rows.matrices();
                // The model's own assumption is no evidence for or against an IMM model.
                let evidence = self.filter.log_likelihood;
                self.filter.update(&residual, &jacobian, &variance, 0.0);
                self.filter.log_likelihood = evidence;
            }
        }
    }

    /// Planar mode fuses nothing in z, roll or pitch (and so vz, wx, wy: the same indices).
    fn pinned(&self, dim: usize) -> bool {
        self.config.planar && (2..5).contains(&dim)
    }

    fn current_pose(&self) -> Isometry3<f64> {
        Isometry3::from_parts(Translation3::from(self.filter.x.position), self.filter.x.q)
    }

    /// A rejected message is a jump in the source when its step from the previous message
    /// disagrees with the filter's motion over that step, confirmed (and the source's frame
    /// re-aligned) when the next message moves on from it. A rejected message that moved on
    /// smoothly from an accepted one means the filter drifted, so it is fused past the gate.
    fn fuse_pose(
        &mut self,
        source: usize,
        pose: &Isometry3<f64>,
        variance: &[f64; 6],
        replaying: bool,
    ) {
        let Some(alignment) = self.sources[source].alignment else {
            self.sources[source].alignment = Some(self.current_pose() * pose.inverse());
            self.remember_pose(source, pose, false);
            return;
        };
        let target = alignment * pose;
        let position_residual = target.translation.vector - self.filter.x.position;
        let rotation_residual = (self.filter.x.q.inverse() * target.rotation).scaled_axis();
        let mut rows = Rows::new(self.filter.dim());
        let mut rotation_rows = Rows::new(self.filter.dim());
        for dim in 0..6 {
            if variance[dim] <= 0.0 || self.pinned(dim) {
                continue;
            }
            if dim < 3 {
                rows.push(position_residual[dim], variance[dim])[dim] = 1.0;
            } else {
                let axis = dim - 3;
                rotation_rows.push(rotation_residual[axis], variance[dim])[6 + axis] = 1.0;
            }
        }
        let rotation_agrees = rotation_rows.is_empty() || self.within_gate(&rotation_rows);
        rows.append(rotation_rows);
        if rows.is_empty() {
            return;
        }
        if self.within_gate(&rows) {
            self.update(&rows);
            self.remember_pose(source, pose, false);
            return;
        }
        let previous = self.sources[source].previous.clone();
        let smooth = previous
            .as_ref()
            .is_some_and(|previous| self.step_agrees(previous, pose, variance));
        match previous {
            Some(previous) if smooth && !previous.jumped => {
                // The filter was overconfident: loosen each disagreeing axis just enough that
                // its residual sits at the gate, so the correction is taken rather than crawled to.
                let gate = self.config.outlier_rejection_allowed_variance;
                for dim in 0..6 {
                    if variance[dim] <= 0.0 || self.pinned(dim) {
                        continue;
                    }
                    let (col, residual) = if dim < 3 {
                        (dim, position_residual[dim])
                    } else {
                        (3 + dim, rotation_residual[dim - 3])
                    };
                    let needed = residual * residual / gate - variance[dim];
                    let current = self.filter.p_cov[(col, col)];
                    if needed > current {
                        self.filter.p_cov[(col, col)] = needed;
                    }
                }
                let (residual, jacobian, variance) = rows.matrices();
                self.filter.update(&residual, &jacobian, &variance, 0.0);
                self.gate_overrides += 1;
                self.remember_pose(source, pose, false);
                if !replaying {
                    warn_throttled!(
                        std::time::Duration::from_secs(10),
                        source = %self.config.odom_sources[source].parent_frame_id,
                        jump_m = position_residual.norm(),
                        jump_rad = rotation_residual.norm(),
                        "pose source outside the gate but moving smoothly: the filter drifted, \
                         so it was fused anyway",
                    );
                }
            }
            Some(_) if smooth => {
                // The fused pose stays put and the source's frame moves: odometry must not jump.
                // Only what disagrees moves, or the heading error of the moment would be locked in.
                let rotation = if rotation_agrees {
                    alignment.rotation
                } else {
                    self.filter.x.q * pose.rotation.inverse()
                };
                let translation = self.filter.x.position - rotation * pose.translation.vector;
                self.sources[source].alignment =
                    Some(Isometry3::from_parts(translation.into(), rotation));
                self.realigned += 1;
                self.remember_pose(source, pose, false);
                if !replaying {
                    warn!(
                        source = %self.config.odom_sources[source].parent_frame_id,
                        jump_m = position_residual.norm(),
                        jump_rad = rotation_residual.norm(),
                        rotation_realigned = !rotation_agrees,
                        "pose source jumped and stayed: re-aligned its frame to the fused pose",
                    );
                }
            }
            _ => {
                self.update(&rows);
                self.remember_pose(source, pose, true);
            }
        }
    }

    fn remember_pose(&mut self, source: usize, pose: &Isometry3<f64>, jumped: bool) {
        let mut fused_variance = [0.0; 6];
        for (dim, value) in fused_variance.iter_mut().enumerate() {
            let col = if dim < 3 { dim } else { 3 + dim };
            *value = self.filter.p_cov[(col, col)];
        }
        self.sources[source].previous = Some(PreviousPose {
            source_pose: *pose,
            fused_pose: self.current_pose(),
            fused_variance,
            jumped,
        });
    }

    /// The source's step since `previous` against the filter's over the same span, both in the
    /// body frame at `previous`, gated like a measurement: two messages' noise plus whatever
    /// the filter's own uncertainty grew by.
    fn step_agrees(
        &self,
        previous: &PreviousPose,
        pose: &Isometry3<f64>,
        variance: &[f64; 6],
    ) -> bool {
        let source_step = previous.source_pose.inverse() * pose;
        let fused_step = previous.fused_pose.inverse() * self.current_pose();
        let translation = source_step.translation.vector - fused_step.translation.vector;
        let rotation = (fused_step.rotation.inverse() * source_step.rotation).scaled_axis();
        let mut distance_sq = 0.0;
        let mut dims = 0;
        for dim in 0..6 {
            if variance[dim] <= 0.0 || self.pinned(dim) {
                continue;
            }
            let col = if dim < 3 { dim } else { 3 + dim };
            let growth = (self.filter.p_cov[(col, col)] - previous.fused_variance[dim]).max(0.0);
            let residual = if dim < 3 {
                translation[dim]
            } else {
                rotation[dim - 3]
            };
            distance_sq += residual * residual / (2.0 * variance[dim] + growth);
            dims += 1;
        }
        let gate = self.config.outlier_rejection_allowed_variance;
        gate <= 0.0 || distance_sq < gate * dims as f64
    }

    /// Not a Kalman update: inverse-variance shares sum to one, so a lone source passes through
    /// exactly.
    fn blend(&mut self, measurement: &Measurement, ts_ns: i64) {
        let source = measurement.source;
        let horizon = ts_ns - (BLEND_ACTIVITY_TIMEOUT_SECONDS * 1.0e9) as i64;
        let returning = self.sources[source].blend_last_ns < horizon;
        self.sources[source].blend_last_ns = ts_ns;
        for dim in 0..6 {
            let variance = measurement.pose_variance[dim];
            self.sources[source].blend_inverse_variance[dim] =
                if variance > 0.0 && !self.pinned(dim) {
                    1.0 / variance
                } else {
                    0.0
                };
        }
        let current = self.current_pose();
        let Some(increment) = measurement.delta else {
            self.sources[source].blend_anchor = Some(current);
            return;
        };
        // Back from a silence the others carried, its delta spans the gap: apply it from where
        // the fused pose stood at its last message, not on top of what the others added since.
        let start = match (returning, self.sources[source].blend_anchor) {
            (false, _) => current,
            (true, Some(anchor)) => anchor,
            (true, None) => {
                self.sources[source].blend_anchor = Some(current);
                return;
            }
        };
        let step = current.inverse() * (start * increment);
        let mut share = [0.0; 6];
        for (dim, share_dim) in share.iter_mut().enumerate() {
            let total: f64 = self
                .sources
                .iter()
                .filter(|other| other.blend_last_ns >= horizon)
                .map(|other| other.blend_inverse_variance[dim])
                .sum();
            if total > 0.0 {
                *share_dim = self.sources[source].blend_inverse_variance[dim] / total;
            }
        }
        let translation = step.translation.vector;
        let rotation = step.rotation.scaled_axis();
        self.filter.x.position += self.filter.x.q
            * Vector3::new(
                share[0] * translation.x,
                share[1] * translation.y,
                share[2] * translation.z,
            );
        self.filter.x.q *= UnitQuaternion::from_scaled_axis(Vector3::new(
            share[3] * rotation.x,
            share[4] * rotation.y,
            share[5] * rotation.z,
        ));
        self.sources[source].blend_anchor = Some(self.current_pose());
    }

    /// Stochastic cloning: the measured motion against the filter's own, from the clone taken
    /// at the source's previous message to now, both still correctable.
    fn fuse_pose_delta(&mut self, slot: usize, delta: &Isometry3<f64>, variance: &[f64; 6]) {
        let Some(clone) = self.filter.x.clones[slot].clone() else {
            return;
        };
        let layout = self.filter.layout;
        let clone_from_world = clone.q.inverse().to_rotation_matrix();
        let predicted_translation = clone_from_world * (self.filter.x.position - clone.position);
        let predicted_rotation = clone.q.inverse() * self.filter.x.q;
        let translation_residual = delta.translation.vector - predicted_translation;
        let rotation_residual = (predicted_rotation.inverse() * delta.rotation).scaled_axis();
        let translation_skew = eskf::skew(&predicted_translation);
        let predicted_rotation_transposed = predicted_rotation.inverse().to_rotation_matrix();
        let clone_position = layout.clone_position_col(slot);
        let clone_rotation = layout.clone_rotation_col(slot);
        let mut rows = Rows::new(self.filter.dim());
        for dim in 0..6 {
            if variance[dim] <= 0.0 || self.pinned(dim) {
                continue;
            }
            if dim < 3 {
                let row = rows.push(translation_residual[dim], variance[dim]);
                for col in 0..3 {
                    row[col] = clone_from_world.matrix()[(dim, col)];
                    row[clone_position + col] = -clone_from_world.matrix()[(dim, col)];
                    row[clone_rotation + col] = translation_skew[(dim, col)];
                }
            } else {
                let axis = dim - 3;
                let row = rows.push(rotation_residual[axis], variance[dim]);
                row[6 + axis] = 1.0;
                for col in 0..3 {
                    row[clone_rotation + col] =
                        -predicted_rotation_transposed.matrix()[(axis, col)];
                }
            }
        }
        if !rows.is_empty() {
            self.update(&rows);
        }
    }

    fn fuse_twist(&mut self, twist: &Twist, variance: &[f64; 6]) {
        let rows = self.twist_rows(twist, variance);
        if !rows.is_empty() {
            self.update(&rows);
        }
    }

    /// Body-frame twist rows, against the rate state or the held gyro sample.
    fn twist_rows(&self, twist: &Twist, variance: &[f64; 6]) -> Rows {
        let body_from_world = self.filter.x.q.inverse().to_rotation_matrix();
        let body_velocity = body_from_world * self.filter.x.velocity;
        let velocity_skew = eskf::skew(&body_velocity);
        let (predicted_angular, angular_correction) = self.filter.angular_velocity();
        let mut rows = Rows::new(self.filter.dim());
        for dim in 0..6 {
            if variance[dim] <= 0.0 || self.pinned(dim) {
                continue;
            }
            if dim < 3 {
                let row = rows.push(twist.linear[dim] - body_velocity[dim], variance[dim]);
                for col in 0..3 {
                    row[3 + col] = body_from_world.matrix()[(dim, col)];
                    row[6 + col] = velocity_skew[(dim, col)];
                }
            } else if let Some((col, sign)) = angular_correction {
                let axis = dim - 3;
                let row = rows.push(twist.angular[axis] - predicted_angular[axis], variance[dim]);
                row[col + axis] = sign;
            }
        }
        rows
    }

    fn within_gate(&self, rows: &Rows) -> bool {
        let (residual, jacobian, variance) = rows.matrices();
        self.filter.within_gate(
            &residual,
            &jacobian,
            &variance,
            self.config.outlier_rejection_allowed_variance,
        )
    }

    fn update(&mut self, rows: &Rows) -> bool {
        let (residual, jacobian, variance) = rows.matrices();
        let accepted = self.filter.update(
            &residual,
            &jacobian,
            &variance,
            self.config.outlier_rejection_allowed_variance,
        );
        if !accepted {
            self.gated += 1;
            // A filter rejecting everything looks exactly like a dead filter otherwise.
            warn_throttled!(
                std::time::Duration::from_secs(10),
                gated = self.gated,
                outlier_rejection_allowed_variance = self.config.outlier_rejection_allowed_variance,
                "measurement rejected by the outlier gate. Raise outlier_rejection_allowed_variance, \
                 or correct the source's pose_variances / twist_variances.",
            );
        }
        accepted
    }

    fn snapshot(&self, event: &mut Event) {
        event.state = self.filter.x.clone();
        event.covariance = self.filter.p_cov.clone();
        event.sources = self.sources.clone();
    }

    fn state_is_publishable(&self, state: &State) -> bool {
        let finite = state.position.iter().all(|v| v.is_finite())
            && state.velocity.iter().all(|v| v.is_finite())
            && state.q.coords.iter().all(|v| v.is_finite());
        let distance = state.position.norm();
        let capped = self.config.max_position_m <= 0.0 || distance <= self.config.max_position_m;
        if !finite || !capped {
            warn_throttled!(
                std::time::Duration::from_secs(10),
                distance,
                max_position_m = self.config.max_position_m,
                "filter state is not publishable; holding the last pose. The filter \
                 diverged: check max_position_m and each source's pose_variances / \
                 twist_variances.",
            );
        }
        finite && capped
    }

    /// The fused estimate at the newest input's stamp, once a publish period has elapsed on
    /// the data clock.
    pub fn maybe_publish(&mut self) -> Option<OdometryEstimate> {
        let latest_ns = self.events.back()?.ts_ns;
        let period = (1.0e9 / self.config.publish_rate.max(1.0)) as i64;
        if latest_ns - self.last_publish_ns < period {
            return None;
        }
        self.publish(latest_ns)
    }

    /// The fused estimate predicted to `now_ns`, for a caller on a timer: the timer sets the
    /// cadence, so output is steady however sparse the sources. None if `now_ns` is not past
    /// the last output, or past the replay buffer, where every source has gone quiet for
    /// longer than it may lag.
    pub fn maybe_publish_at(&mut self, now_ns: i64) -> Option<OdometryEstimate> {
        let latest_ns = self.events.back()?.ts_ns;
        let stale = (now_ns - latest_ns) as f64 > self.config.replay_buffer_seconds * 1.0e9;
        if now_ns <= self.last_publish_ns || stale {
            return None;
        }
        self.publish(now_ns.max(latest_ns))
    }

    fn publish(&mut self, ts_ns: i64) -> Option<OdometryEstimate> {
        let latest = self.events.back()?;
        self.last_publish_ns = ts_ns;
        let mut filter = self.filter.clone();
        filter.x = latest.state.clone();
        filter.p_cov = latest.covariance.clone();
        filter.predict_to(ts_ns);
        if !self.state_is_publishable(&filter.x) {
            return None;
        }
        let state = &filter.x;
        let covariance = &filter.p_cov;

        let body_from_world = state.q.inverse().to_rotation_matrix();
        let body_velocity = body_from_world * state.velocity;
        let (body_angular, angular_correction) = filter.angular_velocity();
        let velocity_world = covariance.view((3, 3), (3, 3)).into_owned();
        let velocity_body =
            body_from_world.matrix() * velocity_world * body_from_world.matrix().transpose();
        let angular_body: Matrix3<f64> = angular_correction
            .map(|(col, _)| covariance.fixed_view::<3, 3>(col, col).into_owned())
            .unwrap_or_default();
        let mut pose_covariance = Matrix6::zeros();
        let mut twist_covariance = Matrix6::zeros();
        for row in 0..3 {
            for col in 0..3 {
                pose_covariance[(row, col)] = covariance[(row, col)];
                pose_covariance[(row, col + 3)] = covariance[(row, col + 6)];
                pose_covariance[(row + 3, col)] = covariance[(row + 6, col)];
                pose_covariance[(row + 3, col + 3)] = covariance[(row + 6, col + 6)];
                twist_covariance[(row, col)] = velocity_body[(row, col)];
                twist_covariance[(row + 3, col + 3)] = angular_body[(row, col)];
            }
        }

        Some(OdometryEstimate {
            timestamp_ns: ts_ns,
            frame_id: self.config.odom_frame_id.clone(),
            child_frame_id: self.config.output_frame_id.clone(),
            pose: Isometry3::from_parts(state.position.into(), state.q),
            pose_covariance,
            twist: Twist {
                linear: body_velocity,
                angular: body_angular,
            },
            twist_covariance,
        })
    }

    pub fn report(&self) {
        info!(
            imu_samples = self.imu_samples,
            measurements = self.measurements,
            gated = self.gated,
            gate_overrides = self.gate_overrides,
            realigned = self.realigned,
            replayed = self.replayed,
            too_late = self.too_late,
            "odometry fusion counters"
        );
        if !self.config.lidars.is_empty() {
            info!(
                slices = self.slices,
                off_clock_scans = self.off_clock_scans,
                matched_points = self.matched_points,
                level_points = self.level_points,
                mount_corrections_deg = ?self
                    .filter
                    .x
                    .mounts
                    .iter()
                    .map(|q| {
                        let (r, p, y) = q.euler_angles();
                        [r.to_degrees(), p.to_degrees(), y.to_degrees()]
                    })
                    .collect::<Vec<_>>(),
                gated_points = self.gated_points,
                map_voxels = self.map.voxel_count(),
                "lidar counters"
            );
        }
    }
}

/// One measurement's rows, built up dimension by dimension.
struct Rows {
    dim: usize,
    rows: Vec<DVector<f64>>,
    residuals: Vec<f64>,
    variances: Vec<f64>,
}

impl Rows {
    fn new(dim: usize) -> Self {
        Self {
            dim,
            rows: Vec::new(),
            residuals: Vec::new(),
            variances: Vec::new(),
        }
    }

    fn push(&mut self, residual: f64, variance: f64) -> &mut DVector<f64> {
        self.residuals.push(residual);
        self.variances.push(variance);
        self.rows.push(DVector::zeros(self.dim));
        self.rows.last_mut().expect("just pushed")
    }

    fn append(&mut self, mut other: Rows) {
        self.rows.append(&mut other.rows);
        self.residuals.append(&mut other.residuals);
        self.variances.append(&mut other.variances);
    }

    fn is_empty(&self) -> bool {
        self.rows.is_empty()
    }

    fn matrices(&self) -> (DVector<f64>, Jacobian, DVector<f64>) {
        let mut jacobian = Jacobian::zeros(self.rows.len(), self.dim);
        for (index, row) in self.rows.iter().enumerate() {
            jacobian.row_mut(index).copy_from(&row.transpose());
        }
        (
            DVector::from_column_slice(&self.residuals),
            jacobian,
            DVector::from_column_slice(&self.variances),
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use nalgebra::Quaternion;

    const GRAVITY: f64 = 9.80665;
    const MS: i64 = 1_000_000;

    fn base_config(use_imu: bool) -> OdometryFusionConfig {
        OdometryFusionConfig {
            odom_frame_id: "odom".to_string(),
            output_frame_id: "base".to_string(),
            publish_rate: 1000.0,
            replay_buffer_seconds: 2.0,
            outlier_rejection_allowed_variance: 9.0,
            max_position_m: 10000.0,
            imus: if use_imu {
                vec![test_imu("imu")]
            } else {
                Vec::new()
            },
            initial_gravity_estimate: GRAVITY,
            initial_stds: InitialStds::default(),
            process_noise: ProcessNoise::default(),
            odom_sources: vec![source("odom", SourceKind::Pose)],
            planar: false,
            zero_velocity_variances: [0.0; 6],
            no_imu_blend: false,
            imm: false,
            lidars: Vec::new(),
            lidar_map: LidarMapConfig::default(),
            imu_as_measurement: false,
        }
    }

    fn test_imu(frame_id: &str) -> ImuConfig {
        ImuConfig {
            frame_id: frame_id.to_string(),
            gyro_noise_density: 0.01,
            gyro_random_walk: 0.001,
            accel_noise_density: 0.1,
            accel_random_walk: 0.01,
            init_samples: 5,
            init_gyro_limit: 0.05,
            ..ImuConfig::default()
        }
    }

    fn source(parent_frame_id: &str, kind: SourceKind) -> SourceConfig {
        let (pose_variances, twist_variances) = match kind {
            SourceKind::Twist => ([0.0; 6], [1e-4; 6]),
            _ => ([1e-4; 6], [0.0; 6]),
        };
        SourceConfig {
            parent_frame_id: parent_frame_id.to_string(),
            child_frame_id: "base".to_string(),
            kind,
            pose_variances,
            twist_variances,
            ..Default::default()
        }
    }

    fn pose_message(parent_frame_id: &str, ts_ns: i64, x: f64) -> OdometryEstimate {
        OdometryEstimate {
            timestamp_ns: ts_ns,
            frame_id: parent_frame_id.to_string(),
            child_frame_id: "base".to_string(),
            pose: Isometry3::translation(x, 0.0, 0.0),
            ..Default::default()
        }
    }

    fn twist_message(parent_frame_id: &str, ts_ns: i64, vx: f64, wz: f64) -> OdometryEstimate {
        OdometryEstimate {
            twist: Twist {
                linear: Vector3::new(vx, 0.0, 0.0),
                angular: Vector3::new(0.0, 0.0, wz),
            },
            ..pose_message(parent_frame_id, ts_ns, 0.0)
        }
    }

    fn imu_message(ts_ns: i64, gyro_z: f64, accel_x: f64) -> ImuSample {
        ImuSample {
            timestamp_ns: ts_ns,
            frame_id: "imu".to_string(),
            angular_velocity: Vector3::new(0.0, 0.0, gyro_z),
            linear_acceleration: Vector3::new(accel_x, 0.0, GRAVITY),
        }
    }

    fn init_imu(core: &mut FusionCore) {
        for sample in 0..5 {
            core.handle_imu(
                &imu_message(sample * 10 * MS, 0.0, 0.0),
                &Isometry3::identity(),
            );
        }
        assert!(core.initialized);
    }

    fn published_position(core: &mut FusionCore) -> Option<Vector3<f64>> {
        core.maybe_publish().map(|msg| msg.pose.translation.vector)
    }

    #[test]
    fn a_pose_source_aligns_on_its_first_message_and_is_followed_after() {
        let mut core = FusionCore::new(base_config(false));
        core.handle_source(&pose_message("odom", 0, 5.0));
        for step in 1..=20 {
            core.handle_source(&pose_message(
                "odom",
                step * 50 * MS,
                5.0 + step as f64 * 0.05,
            ));
        }
        let position = published_position(&mut core).expect("a period has elapsed");
        assert!((position.x - 1.0).abs() < 0.01, "{position}");
    }

    #[test]
    fn a_pose_delta_source_contributes_only_its_motion() {
        let mut config = base_config(false);
        config.odom_sources = vec![source("visual_odom", SourceKind::PoseDelta)];
        let mut core = FusionCore::new(config);
        let mut halfway = 0.0;
        for step in 0..=20 {
            let x = 10.0 + step as f64 * 0.05;
            core.handle_source(&pose_message("visual_odom", step * 50 * MS, x));
            if step == 10 {
                halfway = core.filter.x.position.x;
            }
        }
        let position = published_position(&mut core).expect("a period has elapsed");
        // The first steps fight the prior's zero velocity; once it has converged the motion is the source's.
        assert!(position.x < 1.0, "{position}");
        assert!(
            (position.x - halfway - 0.5).abs() < 0.005,
            "{position} {halfway}"
        );
    }

    #[test]
    fn a_twist_source_is_integrated_through_the_velocity_state() {
        let mut config = base_config(false);
        config.odom_sources = vec![source("wheel", SourceKind::Twist)];
        let mut core = FusionCore::new(config);
        for step in 0..=50 {
            core.handle_source(&twist_message("wheel", step * 20 * MS, 1.0, 0.5));
        }
        let msg = core.maybe_publish().expect("a period has elapsed");
        // The velocity state has to converge first, so the start lags a little.
        assert!(
            (msg.pose.translation.x - 0.95).abs() < 0.1,
            "{}",
            msg.pose.translation.x
        );
        assert!((msg.pose.rotation.euler_angles().2 - 0.5).abs() < 0.05);
        assert!((msg.twist.linear.x - 1.0).abs() < 0.01);
        assert!((msg.twist.angular.z - 0.5).abs() < 0.01);
    }

    #[test]
    fn a_zero_velocity_variance_keeps_a_base_from_sliding_sideways() {
        let lateral_after_one_second = |constraint: f64| {
            let mut config = base_config(false);
            let mut wheel = source("wheel", SourceKind::Twist);
            wheel.twist_variances = [1e-2, 1.0, 0.0, 0.0, 0.0, 1e-2];
            config.odom_sources = vec![wheel];
            config.zero_velocity_variances = [0.0, constraint, 0.0, 0.0, 0.0, 0.0];
            let mut core = FusionCore::new(config);
            for step in 0..=50 {
                let mut msg = twist_message("wheel", step * 20 * MS, 1.0, 0.0);
                msg.twist.linear.y = 0.5;
                core.handle_source(&msg);
            }
            core.maybe_publish()
                .expect("a period has elapsed")
                .pose
                .translation
                .y
        };
        let free = lateral_after_one_second(0.0);
        let constrained = lateral_after_one_second(1e-4);
        assert!(free > 0.3, "{free}");
        assert!(constrained.abs() < 0.01, "{constrained}");
    }

    #[test]
    fn with_an_imu_a_zero_rate_constraint_leaves_the_gyro_bias_alone() {
        let mut config = base_config(true);
        config.odom_sources = vec![source("wheel", SourceKind::Twist)];
        config.zero_velocity_variances = [0.0, 0.0, 0.0, 1e-6, 1e-6, 0.0];
        let mut core = FusionCore::new(config);
        init_imu(&mut core);
        for step in 1..=100 {
            let ts_ns = 40 * MS + step * 10 * MS;
            let mut rocking = imu_message(ts_ns, 0.0, 0.0);
            rocking.angular_velocity.x = 0.2 * if step % 20 < 10 { 1.0 } else { -1.0 };
            core.handle_imu(&rocking, &Isometry3::identity());
            core.handle_source(&twist_message("wheel", ts_ns, 0.0, 0.0));
        }
        let gyro_bias = core.filter.x.biases[0].gyro;
        assert!(gyro_bias.x.abs() < 1e-3, "{gyro_bias:?}");
    }

    #[test]
    fn a_still_twist_source_pins_the_velocity_against_a_noisy_pose_source() {
        let spread = |still_variance: f64| {
            let mut config = base_config(false);
            config.publish_rate = 50.0;
            let mut wheel = source("wheel", SourceKind::Twist);
            wheel.still_variance = still_variance;
            wheel.still_speed = 1e-3;
            wheel.still_angular_speed = 1e-3;
            config.odom_sources = vec![source("odom", SourceKind::Pose), wheel];
            let mut core = FusionCore::new(config);
            let mut xs = Vec::new();
            for step in 0..500 {
                let ts_ns = step * 20 * MS;
                core.handle_twist_source_still(ts_ns);
                if step % 5 == 0 {
                    // Point-LIO-like noise: +-2 mm, alternating.
                    let noise = if step % 10 == 0 { 0.002 } else { -0.002 };
                    core.handle_source(&pose_message("odom", ts_ns, noise));
                }
                if step >= 250 {
                    xs.push(
                        core.maybe_publish_at(ts_ns + MS)
                            .map(|m| (m.pose.translation.x, m.twist.linear.x)),
                    );
                }
            }
            let xs: Vec<(f64, f64)> = xs.into_iter().flatten().collect();
            let range = |values: Vec<f64>| {
                values.iter().cloned().fold(f64::MIN, f64::max)
                    - values.iter().cloned().fold(f64::MAX, f64::min)
            };
            (
                range(xs.iter().map(|x| x.0).collect()),
                range(xs.iter().map(|x| x.1).collect()),
            )
        };
        let (free_x, free_vx) = spread(0.0);
        let (held_x, held_vx) = spread(1e-6);
        assert!(held_vx < free_vx / 50.0, "held {held_vx} free {free_vx}");
        // The motion model keeps its noise, so the pose still averages the source's 4 mm swing down.
        assert!(
            held_x <= free_x && held_x < 0.001,
            "held {held_x} free {free_x}"
        );
    }

    impl FusionCore {
        fn handle_twist_source_still(&mut self, ts_ns: i64) {
            self.handle_source(&twist_message("wheel", ts_ns, 0.0, 0.0));
        }
    }

    #[test]
    fn output_between_sparse_sources_is_predicted_to_the_publish_time() {
        let mut config = base_config(false);
        config.publish_rate = 100.0;
        config.odom_sources = vec![source("wheel", SourceKind::Twist)];
        let mut core = FusionCore::new(config);
        for step in 0..=20 {
            core.handle_source(&twist_message("wheel", step * 100 * MS, 1.0, 0.0));
        }
        let at_source = core.maybe_publish().expect("a period has elapsed");
        let between = core
            .maybe_publish_at(2_050 * MS)
            .expect("a period has elapsed");
        let step = between.pose.translation.x - at_source.pose.translation.x;
        assert!((step - 0.05).abs() < 0.005, "{step}");
        assert_eq!(between.timestamp_ns, 2_050 * MS);
        assert!(
            core.maybe_publish_at(5_000 * MS).is_none(),
            "every source has been silent past the replay buffer"
        );
    }

    #[test]
    fn planar_keeps_z_roll_and_pitch_at_zero_without_and_with_an_imu() {
        for use_imu in [false, true] {
            let mut config = base_config(use_imu);
            config.planar = true;
            config.odom_sources = vec![source("visual_odom", SourceKind::PoseDelta)];
            let mut core = FusionCore::new(config);
            if use_imu {
                init_imu(&mut core);
            }
            let tilt = UnitQuaternion::from_euler_angles(0.1, -0.05, 0.0);
            for step in 0..=20 {
                let ts_ns = 50 * MS + step * 50 * MS;
                if use_imu {
                    core.handle_imu(&imu_message(ts_ns, 0.0, 0.0), &Isometry3::identity());
                }
                let mut msg = pose_message("visual_odom", ts_ns, step as f64 * 0.05);
                msg.pose.translation.z = step as f64 * 0.1;
                msg.pose.rotation = tilt.powf(step as f64);
                core.handle_source(&msg);
            }
            let msg = core.maybe_publish().expect("a period has elapsed");
            let (roll, pitch, _) = msg.pose.rotation.euler_angles();
            assert_eq!(msg.pose.translation.z, 0.0, "use_imu {use_imu}");
            assert!(
                roll.abs() < 1e-12 && pitch.abs() < 1e-12,
                "use_imu {use_imu}"
            );
            assert!(msg.pose.translation.x > 0.5, "use_imu {use_imu}");
        }
    }

    /// Wheels at 50 Hz carry the pose through a silence of the pose source.
    fn wheels_and_pose(pose_at: impl Fn(i64) -> Option<Isometry3<f64>>) -> FusionCore {
        let mut config = base_config(false);
        let mut pose = source("lio", SourceKind::Pose);
        pose.pose_variances = [1e-3; 6];
        config.odom_sources = vec![pose, source("wheel", SourceKind::Twist)];
        let mut core = FusionCore::new(config);
        for step in 0..=300_i64 {
            let ts_ns = step * 20 * MS;
            core.handle_source(&twist_message("wheel", ts_ns, 0.5, 0.0));
            if step % 5 == 0 {
                if let Some(pose) = pose_at(ts_ns) {
                    core.handle_source(&OdometryEstimate {
                        pose,
                        ..pose_message("lio", ts_ns, 0.0)
                    });
                }
            }
        }
        core
    }

    #[test]
    fn the_twist_carries_a_dropout_of_the_pose_source_which_is_taken_back_after() {
        let truth = |ts_ns: i64| 0.5 * ts_ns as f64 / 1.0e9;
        let silent = |ts_ns: i64| (2_000 * MS..4_000 * MS).contains(&ts_ns);
        let core = wheels_and_pose(|ts_ns| {
            (!silent(ts_ns)).then(|| Isometry3::translation(truth(ts_ns), 0.0, 0.0))
        });
        assert_eq!(core.realigned, 0);
        assert_eq!(core.gated, 0);
        assert!((core.filter.x.position.x - truth(6_000 * MS)).abs() < 0.01);
    }

    #[test]
    fn a_pose_source_that_jumps_is_gated_then_realigned_without_moving_the_output() {
        // The source's frame jumps 5 m at 3 s, then turns half a radian about its origin at 4.5 s.
        let jumped = |ts_ns: i64| {
            let truth = Isometry3::translation(0.5 * ts_ns as f64 / 1.0e9, 0.0, 0.0);
            let shift = if ts_ns >= 3_000 * MS { 5.0 } else { 0.0 };
            let turn = if ts_ns >= 4_500 * MS { 0.5 } else { 0.0 };
            let frame = Isometry3::from_parts(
                Translation3::new(shift, 0.0, 0.0),
                UnitQuaternion::from_euler_angles(0.0, 0.0, turn),
            );
            frame * truth
        };
        let core = wheels_and_pose(|ts_ns| Some(jumped(ts_ns)));
        assert_eq!(core.realigned, 2);
        // Each jump is gated once, then the next message confirms it.
        assert_eq!(core.gated, 2);
        assert_eq!(core.gate_overrides, 0);
        // The wheels agree with the motion before and after, so the output never jumps.
        assert!(
            (core.filter.x.position.x - 3.0).abs() < 0.01,
            "{}",
            core.filter.x.position.x
        );
        // Not the half radian.
        assert!(
            core.filter.x.q.angle() < 0.02,
            "{}",
            core.filter.x.q.angle()
        );
        let previous = core.sources[0]
            .previous
            .as_ref()
            .expect("the source has spoken");
        assert!(
            !previous.jumped,
            "the source is fused again after the re-align"
        );
    }

    #[test]
    fn a_push_the_still_wheels_miss_is_taken_from_the_pose_source() {
        // Wheels read zero throughout; the body is pushed 0.6 m over 2 s from t = 4 s.
        let truth = |ts_ns: i64| (0.3 * (ts_ns as f64 / 1.0e9 - 4.0)).clamp(0.0, 0.6);
        let mut config = base_config(false);
        let mut wheel = source("wheel", SourceKind::Twist);
        wheel.still_variance = 1e-6;
        wheel.still_speed = 1e-3;
        wheel.still_angular_speed = 1e-3;
        config.odom_sources = vec![source("lio", SourceKind::Pose), wheel];
        let mut core = FusionCore::new(config);
        for step in 0..=600_i64 {
            let ts_ns = step * 20 * MS;
            core.handle_twist_source_still(ts_ns);
            if step % 5 == 0 {
                core.handle_source(&pose_message("lio", ts_ns, truth(ts_ns)));
            }
        }
        assert_eq!(core.realigned, 0);
        let x = core.filter.x.position.x;
        assert!((x - 0.6).abs() < 0.05, "{x}");
    }

    #[test]
    fn a_one_message_spike_is_gated_without_moving_the_sources_frame() {
        let mut config = base_config(false);
        config.odom_sources = vec![
            source("lio", SourceKind::Pose),
            source("wheel", SourceKind::Twist),
        ];
        let mut core = FusionCore::new(config);
        for step in 0..=100_i64 {
            let ts_ns = step * 20 * MS;
            core.handle_source(&twist_message("wheel", ts_ns, 0.5, 0.0));
            if step % 5 == 0 {
                let spike = if step == 50 { 3.0 } else { 0.0 };
                core.handle_source(&pose_message(
                    "lio",
                    ts_ns,
                    0.5 * ts_ns as f64 / 1.0e9 + spike,
                ));
            }
        }
        assert_eq!((core.gated, core.realigned, core.gate_overrides), (1, 0, 0));
        assert!((core.filter.x.position.x - 1.0).abs() < 0.01);
    }

    #[test]
    fn no_imu_blend_splits_each_axis_by_inverse_variance() {
        let mut config = base_config(false);
        config.no_imu_blend = true;
        let mut wheel = source("wheel", SourceKind::PoseDelta);
        wheel.pose_variances = [3e-4; 6];
        config.odom_sources = vec![source("lio", SourceKind::PoseDelta), wheel];
        let mut core = FusionCore::new(config);
        // lio (1e-4) says 1 m, the wheels (3e-4) 2 m: shares of 3/4 and 1/4.
        for step in 0..=10_i64 {
            let ts_ns = step * 100 * MS;
            core.handle_source(&pose_message("lio", ts_ns, step as f64 * 0.1));
            core.handle_source(&pose_message("wheel", ts_ns + MS, step as f64 * 0.2));
        }
        let x = core.filter.x.position.x;
        assert!((x - 1.25).abs() < 1e-9, "{x}");
    }

    /// A scan of a box room (x -5.17..4.83, y -3.89..4.11, z -1.07..2.93: walls off the map's
    /// voxel grid) from a sensor at `position(t)`, rays over the sphere in a scrambled order over
    /// 100 ms, so each slice spans the room as a non-repetitive Livox pattern does.
    fn room_scan(start_ns: i64, position: impl Fn(i64) -> Vector3<f64>) -> LidarScan {
        let rays = 3000;
        let (low, high) = (Vector3::new(-5.17, -3.89, -1.07), Vector3::new(4.83, 4.11, 2.93));
        let mut scan = LidarScan {
            timestamp_ns: start_ns,
            frame_id: "lidar".into(),
            ..LidarScan::default()
        };
        for ray in 0..rays {
            // Fibonacci sphere: even coverage with no randomness.
            let z = 1.0 - 2.0 * (ray as f64 + 0.5) / rays as f64;
            let azimuth = ray as f64 * std::f64::consts::PI * (3.0 - 5f64.sqrt());
            let radius = (1.0 - z * z).sqrt();
            let direction = Vector3::new(radius * azimuth.cos(), radius * azimuth.sin(), z);
            // 1543 is coprime with 3000, so this visits every slot once, scattered.
            let offset = ((ray as i64 * 1543) % rays as i64 * 100 * MS) / rays as i64;
            let origin = position(start_ns + offset);
            let hit = (0..3)
                .filter(|&axis| direction[axis].abs() > 1e-9)
                .map(|axis| {
                    let wall = if direction[axis] > 0.0 { high[axis] } else { low[axis] };
                    (wall - origin[axis]) / direction[axis]
                })
                .fold(f64::INFINITY, f64::min);
            let point = direction * hit;
            scan.points.push([point.x as f32, point.y as f32, point.z as f32]);
            scan.offsets_ns.push(offset as u32);
        }
        scan
    }

    /// Lidar alone, no IMU and no source: the constant-velocity model carries the pose between
    /// slices and the planes of the room pull it back onto the true path.
    #[test]
    fn lidar_alone_tracks_a_drive_through_a_room() {
        let mut config = base_config(false);
        config.odom_sources.clear();
        config.replay_buffer_seconds = 0.3;
        config.lidars = vec![LidarConfig {
            frame_id: "lidar".into(),
            ..LidarConfig::default()
        }];
        let mut core = FusionCore::new(config);
        let speed = 0.5;
        // Still for a second while the map fills, then 0.5 m/s along x.
        let position = |ts_ns: i64| {
            let moving = ((ts_ns as f64 / 1e9) - 1.0).max(0.0);
            Vector3::new(-2.0 + speed * moving, 0.5, 0.0)
        };
        let start = Vector3::new(-2.0, 0.5, 0.0);
        let mut worst: f64 = 0.0;
        for scan in 0..60 {
            let start_ns = scan * 100 * MS;
            // The sensor reports in its own frame: points relative to where it was then.
            let mut room = room_scan(start_ns, position);
            for (point, offset) in room.points.iter_mut().zip(&room.offsets_ns) {
                let origin = position(start_ns + *offset as i64);
                let world = Vector3::new(point[0] as f64, point[1] as f64, point[2] as f64) + origin;
                let relative = world - origin;
                *point = [relative.x as f32, relative.y as f32, relative.z as f32];
            }
            core.handle_cloud(&room, &Isometry3::identity());
            let now = start_ns + 100 * MS;
            if let Some(out) = core.maybe_publish_at(now) {
                let truth = position(now) - start;
                let error = (out.pose.translation.vector - truth).norm();
                if scan > 15 {
                    worst = worst.max(error);
                }
            }
        }
        let truth = position(6000 * MS) - start;
        assert!(truth.x > 2.0, "the body moved {truth}");
        assert!(worst < 0.05, "worst error {worst:.3} m");
        assert!(core.matched_points > 0);
    }

    #[test]
    fn a_late_message_replays_to_the_same_state_as_one_in_order() {
        let run = |late: bool| {
            let mut config = base_config(false);
            config.odom_sources = vec![
                source("lio", SourceKind::Pose),
                source("wheel", SourceKind::Twist),
            ];
            let mut core = FusionCore::new(config);
            let mut held = None;
            for step in 0..=50_i64 {
                let ts_ns = step * 20 * MS;
                if step % 5 == 0 {
                    let lio = pose_message("lio", ts_ns, 0.3 * ts_ns as f64 / 1.0e9);
                    if late && step == 25 {
                        held = Some(lio);
                    } else {
                        core.handle_source(&lio);
                    }
                }
                core.handle_source(&twist_message("wheel", ts_ns, 0.3, 0.0));
                if step == 30 {
                    if let Some(lio) = held.take() {
                        core.handle_source(&lio);
                    }
                }
            }
            (
                core.filter.x.position,
                core.filter.x.velocity,
                core.replayed,
            )
        };
        let (in_order, in_order_velocity, _) = run(false);
        let (late, late_velocity, replayed) = run(true);
        assert_eq!(replayed, 1);
        assert_eq!(in_order, late);
        assert_eq!(in_order_velocity, late_velocity);
    }

    #[test]
    fn a_source_from_an_unconfigured_frame_is_ignored() {
        let mut core = FusionCore::new(base_config(false));
        core.handle_source(&pose_message("odom", 0, 0.0));
        core.handle_source(&pose_message("odom", 100 * MS, 0.0));
        core.handle_source(&pose_message("stray", 150 * MS, 99.0));
        let position = published_position(&mut core).expect("a period has elapsed");
        assert!(position.x.abs() < 1e-12);
    }

    #[test]
    fn a_matching_parent_with_the_wrong_child_is_ignored() {
        let mut core = FusionCore::new(base_config(false));
        core.handle_source(&pose_message("odom", 0, 0.0));
        core.handle_source(&pose_message("odom", 100 * MS, 0.0));
        let mut wrong_child = pose_message("odom", 150 * MS, 99.0);
        wrong_child.child_frame_id = "some_other_link".to_string();
        core.handle_source(&wrong_child);
        let position = published_position(&mut core).expect("a period has elapsed");
        assert!(position.x.abs() < 1e-12);
    }

    #[test]
    fn an_odom_sources_entry_deserializes_with_defaulted_kind_and_variances() {
        let parsed: Vec<SourceConfig> = serde_json::from_str(
            r#"[{"parent_frame_id": "visual_odom", "child_frame_id": "base"},
                {"parent_frame_id": "wheel", "child_frame_id": "base", "kind": "twist"}]"#,
        )
        .expect("deserializable");
        assert_eq!(parsed[0].kind, SourceKind::PoseDelta);
        assert_eq!(parsed[0].pose_variances, [0.0; 6]);
        assert_eq!(parsed[1].kind, SourceKind::Twist);
    }

    #[test]
    fn a_config_key_this_filter_no_longer_has_is_refused_rather_than_ignored() {
        let parsed: Result<SourceConfig, _> = serde_json::from_str(
            r#"{"parent_frame_id": "odom", "child_frame_id": "base", "absolute": true}"#,
        );
        assert!(parsed.is_err());
        let parsed: Result<OdometryFusionConfig, _> =
            serde_json::from_str(r#"{"blend_weights": [0, 0, 1, 1, 1, 0]}"#);
        assert!(parsed.is_err());
    }

    #[test]
    #[should_panic(expected = "fuses no pose")]
    fn a_twist_source_with_pose_variances_is_a_config_error() {
        let mut config = base_config(false);
        config.odom_sources[0].kind = SourceKind::Twist;
        config.odom_sources[0].twist_variances = [1e-4; 6];
        FusionCore::new(config).handle_source(&pose_message("odom", 0, 0.0));
    }

    #[test]
    fn an_imu_from_an_unconfigured_frame_is_ignored() {
        let mut core = FusionCore::new(base_config(true));
        let mut stray = imu_message(0, 0.0, 0.0);
        stray.frame_id = "some_other_imu".to_string();
        for _ in 0..20 {
            core.handle_imu(&stray, &Isometry3::identity());
        }
        assert!(!core.initialized);
    }

    #[test]
    fn init_waits_until_every_imu_has_its_samples() {
        let mut config = base_config(true);
        config.imus.push(test_imu("imu2"));
        let mut core = FusionCore::new(config);
        for sample in 0..20 {
            core.handle_imu(
                &imu_message(sample * 10 * MS, 0.0, 0.0),
                &Isometry3::identity(),
            );
        }
        assert!(!core.initialized, "the second imu has produced nothing yet");
    }

    #[test]
    fn two_imus_calibrate_their_own_biases_at_init() {
        let mut config = base_config(true);
        config.imus.push(test_imu("imu2"));
        let mut core = FusionCore::new(config);
        for sample in 0..5_i64 {
            let ts_ns = sample * 10 * MS;
            core.handle_imu(&imu_message(ts_ns, 0.0, 0.0), &Isometry3::identity());
            let mut second = imu_message(ts_ns + 5 * MS, 0.0, 0.0);
            second.frame_id = "imu2".to_string();
            second.angular_velocity = Vector3::new(0.02, 0.0, 0.0); // this gyro alone is biased
            core.handle_imu(&second, &Isometry3::identity());
        }
        assert!(core.initialized);
        assert_eq!(core.filter.x.biases.len(), 2);
        assert!(core.filter.x.biases[0].gyro.norm() < 1e-9);
        assert!((core.filter.x.biases[1].gyro.x - 0.02).abs() < 1e-9);
    }

    #[test]
    fn a_nan_pose_is_dropped_rather_than_wedging_the_filter() {
        let mut config = base_config(false);
        config.odom_sources = vec![source("visual_odom", SourceKind::PoseDelta)];
        let mut core = FusionCore::new(config);
        let mut before = 0.0;
        for step in 0..=20 {
            let mut msg = pose_message("visual_odom", step * 50 * MS, 10.0 + step as f64 * 0.05);
            if step == 10 {
                before = core.filter.x.position.x;
                msg.pose.rotation =
                    UnitQuaternion::from_quaternion(Quaternion::new(0.0, 0.0, 0.0, 0.0));
                assert!(msg.pose.rotation.coords.iter().any(|v| v.is_nan()));
            }
            core.handle_source(&msg);
        }
        let position = published_position(&mut core).expect("a period has elapsed");
        // The delta after the NaN spans both steps, so nothing is lost across it.
        assert!(
            (position.x - before - 0.55).abs() < 0.005,
            "{position} {before}"
        );
    }

    #[test]
    fn maybe_publish_waits_for_the_period_and_stamps_odom_to_base() {
        let mut config = base_config(false);
        config.publish_rate = 10.0;
        let mut core = FusionCore::new(config);
        core.handle_source(&pose_message("odom", 0, 0.0));
        assert!(core.maybe_publish().is_none());
        core.handle_source(&pose_message("odom", 50 * MS, 0.0));
        assert!(core.maybe_publish().is_none());
        core.handle_source(&pose_message("odom", 100 * MS, 0.0));
        let msg = core.maybe_publish().expect("a full period has elapsed");
        assert_eq!(msg.frame_id, "odom");
        assert_eq!(msg.child_frame_id, "base");
        assert_eq!(msg.timestamp_ns, 100 * MS);
    }

    #[test]
    fn the_outlier_gate_rejects_a_wild_pose_and_accepts_a_consistent_one() {
        let mut core = FusionCore::new(base_config(true));
        init_imu(&mut core);
        core.handle_source(&pose_message("odom", 50 * MS, 0.0));
        core.handle_source(&pose_message("odom", 100 * MS, 1000.0));
        assert_eq!(core.gated, 1);
        assert!(core.filter.x.position.x.abs() < 1e-6);
        core.handle_source(&pose_message("odom", 150 * MS, 0.01));
        assert_eq!(core.gated, 1);
        assert!(core.filter.x.position.x > 0.005);
    }

    #[test]
    fn a_state_past_the_position_cap_is_held_back_unless_the_cap_is_zero() {
        let mut config = base_config(false);
        config.odom_sources = vec![source("wheel", SourceKind::Twist)];
        config.max_position_m = 0.5;
        let run = |config: OdometryFusionConfig| {
            let mut core = FusionCore::new(config);
            for step in 0..=50 {
                core.handle_source(&twist_message("wheel", step * 20 * MS, 1.0, 0.0));
            }
            core.maybe_publish()
        };
        assert!(run(config.clone()).is_none());
        config.max_position_m = 0.0;
        assert!(run(config).is_some());
    }

    #[test]
    fn the_same_imu_and_source_sequence_replays_bit_identically() {
        let run = || {
            let mut config = base_config(true);
            // Position only: the source reports no rotation, so it must not argue against the yaw.
            config.odom_sources[0].pose_variances = [1e-4, 1e-4, 1e-4, 0.0, 0.0, 0.0];
            config.odom_sources.push(source("wheel", SourceKind::Twist));
            config.odom_sources[1].twist_variances = [1e-2, 1e-2, 0.0, 0.0, 0.0, 0.0];
            let mut core = FusionCore::new(config);
            let mut published = Vec::new();
            for step in 0..60_i64 {
                let ts_ns = step * 10 * MS;
                let moving = step >= 10;
                let gyro_z = if moving { 0.3 } else { 0.0 };
                let accel_x = if moving { 0.5 } else { 0.0 };
                core.handle_imu(&imu_message(ts_ns, gyro_z, accel_x), &Isometry3::identity());
                if step % 10 == 0 {
                    core.handle_source(&pose_message("odom", ts_ns, step as f64 * 0.02));
                }
                if step % 4 == 0 {
                    // Slightly late, so the run replays as well.
                    core.handle_source(&twist_message("wheel", ts_ns - 15 * MS, 0.1, 0.0));
                }
                if let Some(msg) = core.maybe_publish() {
                    let position = msg.pose.translation.vector;
                    published.push((position.x, position.y, position.z, msg.pose.rotation.w));
                }
            }
            assert!(core.replayed > 0);
            published
        };
        let first = run();
        assert!(first.len() > 10);
        // Identical output proves nothing unless the filter actually moved and turned.
        let last = first.last().expect("published above");
        assert!(
            last.0.hypot(last.1) > 0.05,
            "the fused pose barely moved: {last:?}"
        );
        assert!(last.3 < 0.999, "the fused pose barely turned: {last:?}");
        assert_eq!(first, run());
    }

    /// The robot stands still for 5 s, then drives at 0.5 m/s for 5 s, wheels at 50 Hz and a
    /// pose source at 10 Hz with +-2 mm of alternating noise. Returns the published x
    /// (truth subtracted) through the still half and the moving half, and the final x.
    fn still_then_moving(imm: bool) -> (Vec<f64>, Vec<f64>, f64) {
        let mut config = base_config(false);
        config.publish_rate = 50.0;
        config.imm = imm;
        let mut wheel = source("wheel", SourceKind::Twist);
        if !imm {
            wheel.still_variance = 1e-6;
            wheel.still_speed = 1e-3;
            wheel.still_angular_speed = 1e-3;
        }
        config.odom_sources = vec![source("lio", SourceKind::Pose), wheel];
        let truth = |ts_ns: i64| (0.5 * (ts_ns as f64 / 1.0e9 - 5.0)).max(0.0);
        let mut core = FusionCore::new(config);
        let (mut still, mut moving) = (Vec::new(), Vec::new());
        for step in 0..500_i64 {
            let ts_ns = step * 20 * MS;
            let speed = if ts_ns >= 5_000 * MS { 0.5 } else { 0.0 };
            core.handle_source(&twist_message("wheel", ts_ns, speed, 0.0));
            if step % 5 == 0 {
                let noise = if step % 10 == 0 { 0.002 } else { -0.002 };
                core.handle_source(&pose_message("lio", ts_ns, truth(ts_ns) + noise));
            }
            let Some(msg) = core.maybe_publish_at(ts_ns + MS) else {
                continue;
            };
            let error = msg.pose.translation.x - truth(ts_ns + MS);
            if (2_500..5_000).contains(&(step * 20)) {
                still.push(error);
            } else if step * 20 >= 6_000 {
                moving.push(error);
            }
        }
        (still, moving, core.filter.x.position.x)
    }

    fn range(values: &[f64]) -> f64 {
        values.iter().cloned().fold(f64::MIN, f64::max)
            - values.iter().cloned().fold(f64::MAX, f64::min)
    }

    #[test]
    fn imm_removes_the_jitter_at_rest_and_keeps_the_accuracy_in_motion() {
        let (single_still, single_moving, _) = still_then_moving(false);
        let (imm_still, imm_moving, _) = still_then_moving(true);
        // At rest the stopped model's pose is the running mean of the source, so what is left
        // of the jitter shrinks as one over the messages heard.
        assert!(
            range(&imm_still) < range(&single_still) / 4.0,
            "imm {} single {}",
            range(&imm_still),
            range(&single_still)
        );
        let worst = |errors: &[f64]| errors.iter().fold(0.0_f64, |m, e| m.max(e.abs()));
        assert!(
            worst(&imm_moving) <= worst(&single_moving) + 1e-3,
            "imm {} single {}",
            worst(&imm_moving),
            worst(&single_moving)
        );
    }

    #[test]
    fn imm_hands_a_push_the_still_wheels_miss_to_the_moving_model() {
        let truth = |ts_ns: i64| (0.3 * (ts_ns as f64 / 1.0e9 - 4.0)).clamp(0.0, 0.6);
        let mut config = base_config(false);
        config.imm = true;
        config.odom_sources = vec![
            source("lio", SourceKind::Pose),
            source("wheel", SourceKind::Twist),
        ];
        let mut core = FusionCore::new(config);
        for step in 0..=600_i64 {
            let ts_ns = step * 20 * MS;
            core.handle_twist_source_still(ts_ns);
            if step % 5 == 0 {
                core.handle_source(&pose_message("lio", ts_ns, truth(ts_ns)));
            }
        }
        assert_eq!(core.realigned, 0);
        let x = core.filter.x.position.x;
        assert!((x - 0.6).abs() < 0.05, "{x}");
    }

    #[test]
    fn imm_replays_a_late_message_to_the_same_state_as_one_in_order() {
        let run = |late: bool| {
            let mut config = base_config(false);
            config.imm = true;
            config.odom_sources = vec![
                source("lio", SourceKind::Pose),
                source("wheel", SourceKind::Twist),
            ];
            let mut core = FusionCore::new(config);
            let mut held = None;
            for step in 0..=50_i64 {
                let ts_ns = step * 20 * MS;
                if step % 5 == 0 {
                    let lio = pose_message("lio", ts_ns, 0.3 * ts_ns as f64 / 1.0e9);
                    if late && step == 25 {
                        held = Some(lio);
                    } else {
                        core.handle_source(&lio);
                    }
                }
                core.handle_source(&twist_message("wheel", ts_ns, 0.3, 0.0));
                if step == 30 {
                    if let Some(lio) = held.take() {
                        core.handle_source(&lio);
                    }
                }
            }
            (
                core.filter.x.position,
                core.events.back().map(|e| e.imm[1].probability),
            )
        };
        // The late lio lands after the wheel of the same stamp instead of before it.
        let ((in_order, p_in_order), (late, p_late)) = (run(false), run(true));
        assert!((in_order - late).norm() < 1e-9, "{in_order} {late}");
        let (p_in_order, p_late) = (p_in_order.expect("imm"), p_late.expect("imm"));
        assert!((p_in_order - p_late).abs() <= 1e-9 * p_in_order.max(p_late));
    }

    #[test]
    fn imm_follows_a_drifting_visual_source_through_an_arc() {
        const REST: f64 = 3.0;
        // Visual pose deltas at 30 Hz, wheels at 50 Hz with no yaw: a 0.5 m/s, 0.3 rad/s arc
        // after 3 s at rest, the way Alfred fuses heading from vision alone.
        let truth = |ts_ns: i64| {
            let t = (ts_ns as f64 / 1.0e9 - REST).max(0.0);
            let yaw = 0.3 * t;
            let r = 0.5 / 0.3;
            Isometry3::from_parts(
                Translation3::new(r * yaw.sin(), r * (1.0 - yaw.cos()), 0.0),
                UnitQuaternion::from_euler_angles(0.0, 0.0, yaw),
            )
        };
        let run = |imm: bool, v: f64| {
            let mut config = base_config(false);
            config.imm = imm;
            let mut visual = source("visual_odom", SourceKind::PoseDelta);
            visual.pose_variances = [v, v, 0.0, v, v, v];
            let mut wheel = source("wheel", SourceKind::Twist);
            wheel.twist_variances = [0.02, 0.02, 0.0, 0.0, 0.0, 0.0];
            config.odom_sources = vec![visual, wheel];
            let mut core = FusionCore::new(config);
            for step in 0..=3000_i64 {
                let ts_ns = step * MS * 10 / 3;
                if step % 10 == 0 {
                    core.handle_source(&OdometryEstimate {
                        pose: truth(ts_ns),
                        ..pose_message("visual_odom", ts_ns, 0.0)
                    });
                }
                if step % 6 == 0 {
                    let moving = ts_ns as f64 >= REST * 1e9;
                    core.handle_source(&twist_message(
                        "wheel",
                        ts_ns,
                        if moving { 0.5 } else { 0.0 },
                        0.0,
                    ));
                }
            }
            let end = truth(3000 * MS * 10 / 3);
            (core.filter.x.position - end.translation.vector).norm()
        };
        // Nothing absolute corrects a heading lost at the start, so this is where an IMM that
        // carried the stopped model's certain zero rates into the moving one fell apart.
        for variance in [1e-2, 1e-4, 1e-6] {
            let (single, imm) = (run(false, variance), run(true, variance));
            assert!(
                imm < 2.0 * single,
                "variance {variance}: imm {imm} single {single}"
            );
        }
    }
}
