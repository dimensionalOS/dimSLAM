// Copyright 2026 Dimensional Inc.
// SPDX-License-Identifier: Apache-2.0
//
// One error-state Kalman filter fusing odometry sources told apart by their frame_id. With an
// IMU it propagates by inertial integration, without one by a constant-velocity model; every
// source is a measurement of that one state, so the two modes differ only in prediction.

mod eskf;

use std::collections::VecDeque;

use nalgebra::{DVector, Isometry3, Matrix3, Matrix6, Translation3, UnitQuaternion, Vector3};
use serde::{Deserialize, Serialize};
use tracing::{info, warn};

use crate::types::{ImuSample, OdometryEstimate, Twist};
use crate::warn_throttled;
use eskf::{yaw_only, Cov, Filter, ImuBias, ImuInput, Jacobian, State};

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
    /// `pose` only: after rejecting this source this long without a break, the source's frame
    /// is re-aligned to the fused pose rather than kept out for good. 0 never re-aligns.
    pub realign_after_seconds: f64,
    /// Zero-velocity update: a twist at or under `still_speed` (m/s) and `still_angular_speed`
    /// (rad/s) says the body is still: it is fused as zero velocity with `still_variance` in
    /// every axis and, without an IMU, the motion model adds no noise until the source reports
    /// motion, so the pose settles instead of wandering under the other sources' noise. 0 disables.
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
    /// rad/s^2/sqrt(Hz).
    pub angular_acceleration: f64,
}

impl Default for ProcessNoise {
    fn default() -> Self {
        Self {
            linear_acceleration: 1.0,
            angular_acceleration: 1.0,
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
    pub bias: f64,
}

impl Default for InitialStds {
    fn default() -> Self {
        Self {
            position: 0.1,
            velocity: 0.1,
            rotation: 0.05,
            angular_velocity: 0.1,
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
        }
    }
}

#[derive(Clone, Debug)]
struct Measurement {
    source: usize,
    /// Planar-projected in planar mode. For `pose_delta`, marks that this message re-clones.
    pose: Option<Isometry3<f64>>,
    /// `pose_delta`: the motion since the source's previous message.
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
    /// `pose`: when its current unbroken run of gate rejections began.
    rejected_since_ns: Option<i64>,
}

// The snapshot alongside the kind dwarfs any variant, so boxing buys nothing.
#[allow(clippy::large_enum_variant)]
#[derive(Clone, Debug)]
enum EventKind {
    Seed,
    Imu(ImuInput),
    Measurement(Measurement),
}

/// One filter step and the state it left behind, so a late arrival can replay from its slot.
#[derive(Clone, Debug)]
struct Event {
    ts_ns: i64,
    kind: EventKind,
    state: State,
    covariance: Cov,
    sources: Vec<SourceRuntime>,
}

impl Event {
    fn empty(ts_ns: i64, kind: EventKind) -> Self {
        Self {
            ts_ns,
            kind,
            state: State::default(),
            covariance: Cov::zeros(0, 0),
            sources: Vec::new(),
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

pub struct FusionCore {
    config: OdometryFusionConfig,
    filter: Filter,
    sources: Vec<SourceRuntime>,
    /// Per source, the filter's pose-clone slot: `pose_delta` sources that fuse a pose.
    clone_slots: Vec<Option<usize>>,
    initialized: bool,
    imu_runtime: Vec<ImuRuntime>,
    events: VecDeque<Event>,
    /// Per-source last raw pose and stamp. A delta belongs to the message stream, so replay
    /// never redoes it.
    previous_source_pose: Vec<Option<(i64, Isometry3<f64>)>>,
    last_publish_ns: i64,
    imu_samples: u64,
    measurements: u64,
    gated: u64,
    realigned: u64,
    replayed: u64,
    too_late: u64,
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
        Self {
            config,
            filter: Filter::default(),
            sources: vec![SourceRuntime::default(); source_count],
            clone_slots,
            initialized: false,
            imu_runtime,
            events: VecDeque::new(),
            previous_source_pose: vec![None; source_count],
            last_publish_ns: 0,
            imu_samples: 0,
            measurements: 0,
            gated: 0,
            realigned: 0,
            replayed: 0,
            too_late: 0,
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
        {
            let runtime = &mut self.imu_runtime[index];
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
        let input = ImuInput {
            imu: index,
            gyro,
            accel,
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
        if source_config.kind == SourceKind::PoseDelta {
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

        let kind = source_config.kind;
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
        if inserted && kind == SourceKind::PoseDelta {
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
                source.realign_after_seconds >= 0.0,
                "source {name:?} needs realign_after_seconds at or above zero"
            );
        }
        if !self.config.use_imu() {
            let noise = &self.config.process_noise;
            assert!(
                noise.linear_acceleration > 0.0 && noise.angular_acceleration > 0.0,
                "without an imu, process_noise needs both densities above zero, or the filter \
                 grows overconfident and gates every source out"
            );
        }
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
            })
            .collect();
        self.filter.process_noise = self.config.process_noise;
        self.filter.gravity = self.config.initial_gravity_estimate;
        self.filter.planar = self.config.planar;
        self.filter.layout.clones = self.clone_slots.iter().flatten().count();
        let gyro_bias_norms: Vec<f64> = biases.iter().map(|bias| bias.gyro.norm()).collect();
        self.filter.init(
            timestamp_ns,
            world_from_body,
            biases,
            input,
            &self.config.initial_stds,
        );
        self.initialized = true;
        self.last_publish_ns = timestamp_ns;
        let mut seed = Event::empty(timestamp_ns, EventKind::Seed);
        self.snapshot(&mut seed);
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
        let horizon = self.events.back().expect("nonempty").ts_ns
            - (self.config.replay_buffer_seconds * 1.0e9) as i64;
        while self.events.len() > 1 && self.events.front().expect("nonempty").ts_ns < horizon {
            self.events.pop_front();
        }
        true
    }

    fn process_at(&mut self, index: usize) {
        {
            let previous = &self.events[index - 1];
            self.filter.x = previous.state.clone();
            self.filter.p_cov = previous.covariance.clone();
            self.sources = previous.sources.clone();
        }
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
            EventKind::Imu(input) => self.filter.x.input = Some(input),
            EventKind::Measurement(measurement) => self.apply(&measurement, ts_ns, replaying),
        }
        let mut event = std::mem::replace(
            &mut self.events[index],
            Event::empty(ts_ns, EventKind::Seed),
        );
        self.snapshot(&mut event);
        self.events[index] = event;
    }

    fn apply(&mut self, measurement: &Measurement, ts_ns: i64, replaying: bool) {
        let source = measurement.source;
        match self.config.odom_sources[source].kind {
            SourceKind::Pose => {
                if let Some(pose) = &measurement.pose {
                    self.fuse_pose(source, pose, &measurement.pose_variance, ts_ns, replaying);
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
            self.filter.x.still = still;
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
                self.filter.update(&residual, &jacobian, &variance, 0.0);
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

    fn fuse_pose(
        &mut self,
        source: usize,
        pose: &Isometry3<f64>,
        variance: &[f64; 6],
        ts_ns: i64,
        replaying: bool,
    ) {
        let Some(alignment) = self.sources[source].alignment else {
            self.sources[source].alignment = Some(self.current_pose() * pose.inverse());
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
        if self.update(&rows) {
            self.sources[source].rejected_since_ns = None;
            return;
        }
        let realign_after = self.config.odom_sources[source].realign_after_seconds;
        let since = *self.sources[source].rejected_since_ns.get_or_insert(ts_ns);
        if realign_after > 0.0 && (ts_ns - since) as f64 >= realign_after * 1.0e9 {
            // The fused pose stays put and the source's frame moves: odometry must not jump. Only
            // what disagrees moves, or the heading error of the moment would be locked in.
            let rotation = if rotation_agrees {
                alignment.rotation
            } else {
                self.filter.x.q * pose.rotation.inverse()
            };
            let translation = self.filter.x.position - rotation * pose.translation.vector;
            self.sources[source].alignment =
                Some(Isometry3::from_parts(translation.into(), rotation));
            self.sources[source].rejected_since_ns = None;
            self.realigned += 1;
            if !replaying {
                warn!(
                    source = %self.config.odom_sources[source].parent_frame_id,
                    jump_m = position_residual.norm(),
                    jump_rad = rotation_residual.norm(),
                    rotation_realigned = !rotation_agrees,
                    realign_after_seconds = realign_after,
                    "pose source rejected for too long: re-aligned its frame to the fused pose",
                );
            }
        }
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
            realigned = self.realigned,
            replayed = self.replayed,
            too_late = self.too_late,
            "odometry fusion counters"
        );
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
            realign_after_seconds: 1.0,
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
    fn a_still_twist_source_holds_the_pose_against_a_noisy_pose_source() {
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
                    xs.push(core.maybe_publish_at(ts_ns + MS).map(|m| m.pose.translation.x));
                }
            }
            let xs: Vec<f64> = xs.into_iter().flatten().collect();
            xs.iter().cloned().fold(f64::MIN, f64::max) - xs.iter().cloned().fold(f64::MAX, f64::min)
        };
        let free = spread(0.0);
        let held = spread(1e-6);
        assert!(held < free / 5.0, "held {held} free {free}");
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
        // A 5 m jump in position at 3 s, then one of half a radian in heading at 4.5 s.
        let jumped = |ts_ns: i64| {
            let x = 0.5 * ts_ns as f64 / 1.0e9;
            let shift = if ts_ns >= 3_000 * MS { 5.0 } else { 0.0 };
            let turn = if ts_ns >= 4_500 * MS { 0.5 } else { 0.0 };
            Isometry3::from_parts(
                Translation3::new(x + shift, 0.0, 0.0),
                UnitQuaternion::from_euler_angles(0.0, 0.0, turn),
            )
        };
        let core = wheels_and_pose(|ts_ns| Some(jumped(ts_ns)));
        assert_eq!(core.realigned, 2);
        // The wheels agree with the motion before and after, so the output never jumps.
        assert!(
            (core.filter.x.position.x - 3.0).abs() < 0.01,
            "{}",
            core.filter.x.position.x
        );
        // Not the half radian: only the wheels' own heading drift through the rejected second.
        assert!(
            core.filter.x.q.angle() < 0.02,
            "{}",
            core.filter.x.q.angle()
        );
        let reaccepted = core.sources[0].rejected_since_ns.is_none();
        assert!(reaccepted, "the source is fused again after the re-align");
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
        let mut config = base_config(true);
        config.odom_sources[0].realign_after_seconds = 0.0;
        let mut core = FusionCore::new(config);
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
}
