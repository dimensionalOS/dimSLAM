// Copyright 2026 Dimensional Inc.
// SPDX-License-Identifier: Apache-2.0
//
// Error-state Kalman filter fusing IMU propagation with odometry sources told apart by their
// frame_id. Without an IMU there is no process model: see `blend`.

mod eskf;

use std::collections::{HashMap, VecDeque};

use nalgebra::{DVector, Isometry3, Matrix6, Translation3, UnitQuaternion, Vector3};
use serde::{Deserialize, Serialize};
use tracing::info;

use crate::types::{ImuSample, OdometryEstimate, Twist};
use crate::warn_throttled;
use eskf::{gyro_bias_col, Cov, Filter, ImuBias, Jacobian, State};

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

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
#[serde(default)]
pub struct SourceConfig {
    /// The transform this source's estimates carry. Both halves are needed: two sources can
    /// share a parent.
    pub parent_frame_id: String,
    pub child_frame_id: String,
    /// [x y z roll pitch yaw]: <0 message covariance, 0 drop, >0 fixed.
    /// A drifting source's covariance describes accumulated drift, not the delta: prefer a fixed value.
    pub pose_variances: [f64; 6],
    /// [vx vy vz wx wy wz], same convention, body frame.
    pub twist_variances: [f64; 6],
}

/// These all describe the physical IMU rather than the filter itself, so they live together
/// in one object.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(default)]
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

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(default)]
pub struct OdometryFusionConfig {
    pub odom_frame_id: String,
    pub output_frame_id: String,
    /// Output cadence; the filter itself runs at the IMU rate.
    pub publish_rate: f64,
    /// How far back a late measurement can reach before it is dropped.
    pub replay_buffer_seconds: f64,
    /// Outlier gate in variance units: reject a measurement whose squared Mahalanobis distance
    /// per measurement dimension exceeds this multiple of the expected noise variance.
    /// Smaller is more aggressive, 0 is no gate.
    pub outlier_rejection_allowed_variance: f64,
    /// Gate on the filter's own state, which can run away with no bad measurement; 0 disables.
    pub max_position_m: f64,
    /// One entry per IMU, told apart by the frame_id its samples carry. Each keeps its own
    /// bias states in the filter; interleaved samples each integrate the slice of time since
    /// the previous sample of any IMU. Empty disables the Kalman machinery entirely: the
    /// filter seeds level from the first source message and holds pose between them (see
    /// `blend`).
    pub imus: Vec<ImuConfig>,
    /// m/s^2. Seeds `filter.gravity` at init; a future ZUPT is meant to refine it from there.
    pub initial_gravity_estimate: f64,
    pub initial_stds: InitialStds,
    /// One entry per external odometry stream, told apart by the transform its estimates carry.
    /// The no-IMU blend sums in list order, so the order is part of the config.
    pub odom_sources: Vec<SourceConfig>,
    /// Virtual zero-twist [vx vy vz wx wy wz] fused after each source update (needs an IMU):
    /// >0 pins that dimension to zero with this variance (e.g. non-holonomic vy, vz), 0 frees it.
    pub constraint_twist_variances: [f64; 6],
}

/// 1-sigma uncertainty the filter starts with, per state block. The bias std applies to
/// every IMU's gyro and accel bias blocks.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(default)]
pub struct InitialStds {
    pub position: f64,
    pub velocity: f64,
    pub rotation: f64,
    pub bias: f64,
}

impl Default for InitialStds {
    fn default() -> Self {
        Self {
            position: 0.1,
            velocity: 0.1,
            rotation: 0.05,
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
            odom_sources: Vec::new(),
            constraint_twist_variances: [0.0; 6],
        }
    }
}

#[derive(Clone, Debug)]
struct Measurement {
    source: usize,
    delta: Option<Isometry3<f64>>,
    pose_variance: [f64; 6], // 0 means dropped
    twist_variance: [f64; 6],
    linear: Vector3<f64>,
    angular: Vector3<f64>,
}

/// A source keeps its share of the no-IMU blend until it has been silent for the timeout.
#[derive(Clone, Debug, Default)]
struct SourceActivity {
    last_ns: i64,
    inverse_variance: [f64; 6],
}

const SOURCE_ACTIVITY_TIMEOUT_SECONDS: f64 = 0.5;

// The snapshot alongside the kind dwarfs any variant, so boxing buys nothing.
#[allow(clippy::large_enum_variant)]
#[derive(Clone, Debug)]
enum EventKind {
    Seed,
    Imu {
        imu: usize,
        gyro: Vector3<f64>,
        accel: Vector3<f64>,
    },
    Measurement(Measurement),
}

/// One filter step and the state it left behind, so a late arrival can replay from its slot.
#[derive(Clone, Debug)]
struct Event {
    ts_ns: i64,
    kind: EventKind,
    state: State,
    covariance: Cov,
    anchors: Vec<Option<Isometry3<f64>>>,
    activity: Vec<SourceActivity>,
    last_imu_ns: i64,
    last_gyro: Vector3<f64>,
    /// Which IMU `last_gyro` came from, so its bias is the one subtracted.
    last_gyro_imu: usize,
}

impl Event {
    fn empty(ts_ns: i64, kind: EventKind) -> Self {
        Self {
            ts_ns,
            kind,
            state: State::default(),
            covariance: Cov::identity(9, 9),
            anchors: Vec::new(),
            activity: Vec::new(),
            last_imu_ns: 0,
            last_gyro: Vector3::zeros(),
            last_gyro_imu: 0,
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
    initialized: bool,
    imu_runtime: Vec<ImuRuntime>,
    events: VecDeque<Event>,
    anchors: Vec<Option<Isometry3<f64>>>,
    activity: Vec<SourceActivity>,
    last_imu_ns: i64,
    last_gyro: Vector3<f64>,
    last_gyro_imu: usize,
    /// Per-source last raw pose. A delta belongs to the message stream, so replay never redoes it.
    previous_source_pose: HashMap<usize, Isometry3<f64>>,
    last_publish_ns: i64,
    imu_samples: u64,
    measurements: u64,
    gated: u64,
    replayed: u64,
    too_late: u64,
}

impl FusionCore {
    pub fn new(config: OdometryFusionConfig) -> Self {
        let imu_runtime = vec![ImuRuntime::default(); config.imus.len()];
        Self {
            config,
            filter: Filter::default(),
            initialized: false,
            imu_runtime,
            events: VecDeque::new(),
            anchors: Vec::new(),
            activity: Vec::new(),
            last_imu_ns: 0,
            last_gyro: Vector3::zeros(),
            last_gyro_imu: 0,
            previous_source_pose: HashMap::new(),
            last_publish_ns: 0,
            imu_samples: 0,
            measurements: 0,
            gated: 0,
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
                let world_from_body = UnitQuaternion::rotation_between(&combined, &Vector3::z())
                    .unwrap_or_default();
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
                self.initialize(ts_ns, world_from_body, biases);
            }
            return;
        }
        self.insert(Event::empty(
            ts_ns,
            EventKind::Imu {
                imu: index,
                gyro,
                accel,
            },
        ));
        self.imu_samples += 1;
    }

    pub fn handle_source(&mut self, msg: &OdometryEstimate) {
        if !self.initialized {
            if self.config.use_imu() {
                return;
            }
            self.check_config();
            self.initialize(msg.timestamp_ns, UnitQuaternion::identity(), Vec::new());
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

        let pose_wanted = pose_variance.iter().any(|v| *v > 0.0);

        let mut measurement = Measurement {
            source,
            delta: None,
            pose_variance,
            twist_variance,
            linear: msg.twist.linear,
            angular: msg.twist.angular,
        };
        // A caller building a UnitQuaternion from a zeroed message gets NaN, not an error, and
        // one NaN reaching the state wedges the filter for good: nothing downstream recovers.
        let pose_finite = msg.pose.translation.vector.iter().all(|v| v.is_finite())
            && msg.pose.rotation.coords.iter().all(|v| v.is_finite());
        if pose_wanted && pose_finite {
            if let Some(previous) = self.previous_source_pose.get(&source) {
                measurement.delta = Some(previous.inverse() * msg.pose);
            }
            self.previous_source_pose.insert(source, msg.pose);
        }

        self.insert(Event::empty(
            msg.timestamp_ns,
            EventKind::Measurement(measurement),
        ));
        self.measurements += 1;
    }

    fn check_config(&self) {
        for source in &self.config.odom_sources {
            assert!(
                !source.parent_frame_id.trim().is_empty()
                    && !source.child_frame_id.trim().is_empty(),
                "an `odom_sources` entry needs both parent_frame_id and child_frame_id set"
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
        self.filter.gravity = self.config.initial_gravity_estimate;
        let gyro_bias_norms: Vec<f64> = biases.iter().map(|bias| bias.gyro.norm()).collect();
        self.filter.init(
            world_from_body,
            biases,
            self.config.initial_stds.position,
            self.config.initial_stds.velocity,
            self.config.initial_stds.rotation,
            self.config.initial_stds.bias,
        );
        self.anchors = vec![None; self.config.odom_sources.len()];
        self.activity = vec![SourceActivity::default(); self.config.odom_sources.len()];
        self.initialized = true;
        self.last_publish_ns = timestamp_ns;
        let mut seed = Event::empty(timestamp_ns, EventKind::Seed);
        self.snapshot(&mut seed);
        self.events.push_back(seed);
        info!(
            gyro_biases = ?gyro_bias_norms,
            use_imu = self.config.use_imu(),
            "odometry_fusion initialized"
        );
    }

    fn insert(&mut self, event: Event) {
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
            return;
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
    }

    fn process_at(&mut self, index: usize) {
        {
            let previous = &self.events[index - 1];
            self.filter.x = previous.state.clone();
            self.filter.p_cov = previous.covariance.clone();
            self.anchors = previous.anchors.clone();
            self.activity = previous.activity.clone();
            self.last_imu_ns = previous.last_imu_ns;
            self.last_gyro = previous.last_gyro;
            self.last_gyro_imu = previous.last_gyro_imu;
        }
        let ts_ns = self.events[index].ts_ns;
        let kind = self.events[index].kind.clone();
        match kind {
            EventKind::Seed => {}
            EventKind::Imu { imu, gyro, accel } => {
                let dt = (ts_ns - self.last_imu_ns) as f64 / 1.0e9;
                if dt > 0.0 && dt < 1.0 {
                    self.filter.propagate(dt, imu, &gyro, &accel);
                }
                self.last_imu_ns = ts_ns;
                self.last_gyro = gyro;
                self.last_gyro_imu = imu;
            }
            EventKind::Measurement(measurement) => {
                if self.config.use_imu() {
                    self.apply(&measurement);
                } else {
                    self.blend(&measurement, ts_ns);
                }
            }
        }
        let mut event = std::mem::replace(&mut self.events[index], Event::empty(ts_ns, EventKind::Seed));
        self.snapshot(&mut event);
        self.events[index] = event;
    }

    fn apply(&mut self, measurement: &Measurement) {
        let mut rows: Vec<DVector<f64>> = Vec::new();
        let mut residuals: Vec<f64> = Vec::new();
        let mut variances: Vec<f64> = Vec::new();

        let target = if let Some(delta) = measurement.delta {
            match self.anchors[measurement.source] {
                Some(anchor) => Some(anchor * delta),
                None => {
                    // First delta from this source: adopt the filter pose, fuse nothing.
                    self.anchor(measurement.source);
                    return;
                }
            }
        } else {
            None
        };
        if let Some(target) = target {
            let position_residual = target.translation.vector - self.filter.x.position;
            let rotation_residual = (self.filter.x.q.inverse() * target.rotation).scaled_axis();
            for dim in 0..3 {
                if measurement.pose_variance[dim] > 0.0 {
                    let mut row = DVector::zeros(self.filter.dim());
                    row[dim] = 1.0;
                    rows.push(row);
                    residuals.push(position_residual[dim]);
                    variances.push(measurement.pose_variance[dim]);
                }
            }
            for dim in 0..3 {
                if measurement.pose_variance[3 + dim] > 0.0 {
                    let mut row = DVector::zeros(self.filter.dim());
                    row[6 + dim] = 1.0;
                    rows.push(row);
                    residuals.push(rotation_residual[dim]);
                    variances.push(measurement.pose_variance[3 + dim]);
                }
            }
        }

        self.add_twist_rows(
            &measurement.linear,
            &measurement.angular,
            &measurement.twist_variance,
            &mut rows,
            &mut residuals,
            &mut variances,
        );
        let accepted = self.update(&rows, &residuals, &variances);
        if accepted && target.is_some() {
            // Anchoring to the corrected pose, not each source's own chain, keeps drifters bounded.
            self.anchor(measurement.source);
        }

        let constraint_variance = self.config.constraint_twist_variances;
        let mut constraint_rows = Vec::new();
        let mut constraint_residuals = Vec::new();
        let mut constraint_variances = Vec::new();
        self.add_twist_rows(
            &Vector3::zeros(),
            &Vector3::zeros(),
            &constraint_variance,
            &mut constraint_rows,
            &mut constraint_residuals,
            &mut constraint_variances,
        );
        self.update(
            &constraint_rows,
            &constraint_residuals,
            &constraint_variances,
        );
    }

    /// Not a Kalman update: inverse-variance shares sum to one, so a lone source passes through exactly.
    fn blend(&mut self, measurement: &Measurement, ts_ns: i64) {
        let activity = &mut self.activity[measurement.source];
        activity.last_ns = ts_ns;
        for dim in 0..6 {
            activity.inverse_variance[dim] = if measurement.pose_variance[dim] > 0.0 {
                1.0 / measurement.pose_variance[dim]
            } else {
                0.0
            };
        }

        let Some(increment) = measurement.delta else {
            return;
        };

        let horizon = ts_ns - (SOURCE_ACTIVITY_TIMEOUT_SECONDS * 1.0e9) as i64;
        let mut share = [0.0; 6];
        for (dim, share_dim) in share.iter_mut().enumerate() {
            let total: f64 = self
                .activity
                .iter()
                .filter(|a| a.last_ns >= horizon)
                .map(|a| a.inverse_variance[dim])
                .sum();
            if total > 0.0 {
                *share_dim = self.activity[measurement.source].inverse_variance[dim] / total;
            }
        }

        let translation = increment.translation.vector;
        let rotation = increment.rotation.scaled_axis();
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
    }

    /// Body-frame twist rows. Angular rate is not a state: it is predicted from the last gyro
    /// sample, so the gyro-bias row belongs to the IMU that sample came from.
    fn add_twist_rows(
        &self,
        linear: &Vector3<f64>,
        angular: &Vector3<f64>,
        variance: &[f64; 6],
        rows: &mut Vec<DVector<f64>>,
        residuals: &mut Vec<f64>,
        variances: &mut Vec<f64>,
    ) {
        let body_from_world = self.filter.x.q.inverse().to_rotation_matrix();
        let body_velocity = body_from_world * self.filter.x.velocity;
        let velocity_skew = eskf::skew(&body_velocity);
        for dim in 0..3 {
            if variance[dim] > 0.0 {
                let mut row = DVector::zeros(self.filter.dim());
                for col in 0..3 {
                    row[3 + col] = body_from_world.matrix()[(dim, col)];
                    row[6 + col] = velocity_skew[(dim, col)];
                }
                rows.push(row);
                residuals.push(linear[dim] - body_velocity[dim]);
                variances.push(variance[dim]);
            }
        }
        let gyro_bias = self.filter.x.biases[self.last_gyro_imu].gyro;
        let predicted_angular = self.last_gyro - gyro_bias;
        for dim in 0..3 {
            if variance[3 + dim] > 0.0 {
                let mut row = DVector::zeros(self.filter.dim());
                row[gyro_bias_col(self.last_gyro_imu) + dim] = -1.0;
                rows.push(row);
                residuals.push(angular[dim] - predicted_angular[dim]);
                variances.push(variance[3 + dim]);
            }
        }
    }

    fn update(&mut self, rows: &[DVector<f64>], residuals: &[f64], variances: &[f64]) -> bool {
        if rows.is_empty() {
            return false;
        }
        let residual = DVector::from_row_slice(residuals);
        let variance = DVector::from_row_slice(variances);
        let mut jacobian = Jacobian::zeros(rows.len(), self.filter.dim());
        for (index, row) in rows.iter().enumerate() {
            jacobian.row_mut(index).copy_from(&row.transpose());
        }
        let accepted = self.filter.update(
            &residual,
            &jacobian,
            &variance,
            // The filter gates in standard deviations; the config speaks variance.
            self.config.outlier_rejection_allowed_variance.sqrt(),
        );
        if !accepted {
            self.gated += 1;
            // A filter rejecting everything looks exactly like a dead filter otherwise.
            warn_throttled!(
                std::time::Duration::from_secs(10),
                gated = self.gated,
                outlier_rejection_allowed_variance = self.config.outlier_rejection_allowed_variance,
                "measurement rejected by the outlier gate. Raise outlier_rejection_allowed_variance, \
                 or \
                 correct the source's pose_variances / twist_variances.",
            );
        }
        accepted
    }

    fn anchor(&mut self, source: usize) {
        self.anchors[source] = Some(Isometry3::from_parts(
            Translation3::from(self.filter.x.position),
            self.filter.x.q,
        ));
    }

    fn snapshot(&self, event: &mut Event) {
        event.state = self.filter.x.clone();
        event.covariance = self.filter.p_cov.clone();
        event.anchors = self.anchors.clone();
        event.activity = self.activity.clone();
        event.last_imu_ns = self.last_imu_ns;
        event.last_gyro = self.last_gyro;
        event.last_gyro_imu = self.last_gyro_imu;
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

    pub fn maybe_publish(&mut self) -> Option<OdometryEstimate> {
        let latest = self.events.back()?;
        let period = (1.0e9 / self.config.publish_rate.max(1.0)) as i64;
        if latest.ts_ns - self.last_publish_ns < period {
            return None;
        }
        self.last_publish_ns = latest.ts_ns;
        let state = latest.state.clone();
        if !self.state_is_publishable(&state) {
            return None;
        }
        let covariance = latest.covariance.clone();
        let last_gyro = latest.last_gyro;
        let last_gyro_imu = latest.last_gyro_imu;
        let ts_ns = latest.ts_ns;

        let body_from_world = state.q.inverse().to_rotation_matrix();
        let body_velocity = body_from_world * state.velocity;
        // The predicted rate is the last gyro sample minus that IMU's own bias; without an
        // IMU there is no rate estimate at all.
        let gyro_bias = state
            .biases
            .get(last_gyro_imu)
            .map(|bias| bias.gyro)
            .unwrap_or_default();
        let body_angular = last_gyro - gyro_bias;

        let velocity_world = covariance.view((3, 3), (3, 3)).into_owned();
        let velocity_body =
            body_from_world.matrix() * velocity_world * body_from_world.matrix().transpose();
        let mut pose_covariance = Matrix6::zeros();
        let mut twist_covariance = Matrix6::zeros();
        let angular_block = (!state.biases.is_empty()).then(|| gyro_bias_col(last_gyro_imu));
        for row in 0..3 {
            for col in 0..3 {
                pose_covariance[(row, col)] = covariance[(row, col)];
                pose_covariance[(row, col + 3)] = covariance[(row, col + 6)];
                pose_covariance[(row + 3, col)] = covariance[(row + 6, col)];
                pose_covariance[(row + 3, col + 3)] = covariance[(row + 6, col + 6)];
                twist_covariance[(row, col)] = velocity_body[(row, col)];
                if let Some(block) = angular_block {
                    twist_covariance[(row + 3, col + 3)] = covariance[(block + row, block + col)];
                }
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
            replayed = self.replayed,
            too_late = self.too_late,
            "odometry fusion counters"
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use nalgebra::Quaternion;

    const GRAVITY: f64 = 9.80665;

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
            odom_sources: one_source("odom"),
            constraint_twist_variances: [0.0; 6],
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

    fn one_source(parent_frame_id: &str) -> Vec<SourceConfig> {
        vec![SourceConfig {
            parent_frame_id: parent_frame_id.to_string(),
            child_frame_id: "base".to_string(),
            pose_variances: [1e-4; 6],
            twist_variances: [0.0; 6],
        }]
    }

    fn source_message(parent_frame_id: &str, ts_ns: i64, x: f64) -> OdometryEstimate {
        OdometryEstimate {
            timestamp_ns: ts_ns,
            frame_id: parent_frame_id.to_string(),
            child_frame_id: "base".to_string(),
            pose: Isometry3::translation(x, 0.0, 0.0),
            ..Default::default()
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

    fn identity_mount() -> Isometry3<f64> {
        Isometry3::identity()
    }

    fn published_position(core: &mut FusionCore) -> Option<Vector3<f64>> {
        core.maybe_publish().map(|msg| msg.pose.translation.vector)
    }

    #[test]
    fn a_single_source_without_an_imu_drives_the_fused_pose() {
        let mut core = FusionCore::new(base_config(false));
        core.handle_source(&source_message("odom", 0, 0.0));
        core.handle_source(&source_message("odom", NS_PER_SEC / 10, 1.0));
        let position = published_position(&mut core).expect("a period has elapsed");
        assert!((position.x - 1.0).abs() < 1e-12);
    }

    #[test]
    fn a_drifting_source_contributes_only_its_delta() {
        let mut config = base_config(false);
        config.odom_sources = one_source("visual_odom");
        let mut core = FusionCore::new(config);
        core.handle_source(&source_message("visual_odom", 0, 10.0));
        core.handle_source(&source_message("visual_odom", NS_PER_SEC / 10, 10.5));
        let position = published_position(&mut core).expect("a period has elapsed");
        assert!((position.x - 0.5).abs() < 1e-12);
    }

    #[test]
    fn a_source_from_an_unconfigured_frame_is_ignored() {
        let mut core = FusionCore::new(base_config(false));
        core.handle_source(&source_message("odom", 0, 0.0));
        core.handle_source(&source_message("odom", NS_PER_SEC / 10, 1.0));
        core.handle_source(&source_message("stray", NS_PER_SEC / 5, 99.0));
        let position = published_position(&mut core).expect("a period has elapsed");
        assert!((position.x - 1.0).abs() < 1e-12);
    }

    #[test]
    fn a_matching_parent_with_the_wrong_child_is_ignored() {
        let mut core = FusionCore::new(base_config(false));
        core.handle_source(&source_message("odom", 0, 0.0));
        core.handle_source(&source_message("odom", NS_PER_SEC / 10, 1.0));
        let mut wrong_child = source_message("odom", NS_PER_SEC / 5, 99.0);
        wrong_child.child_frame_id = "some_other_link".to_string();
        core.handle_source(&wrong_child);
        let position = published_position(&mut core).expect("a period has elapsed");
        assert!((position.x - 1.0).abs() < 1e-12);
    }

    #[test]
    fn an_odom_sources_entry_deserializes_with_defaulted_variances() {
        let parsed: Vec<SourceConfig> = serde_json::from_str(
            r#"[{"parent_frame_id": "visual_odom", "child_frame_id": "base"}]"#,
        )
        .expect("deserializable");
        assert_eq!(parsed[0].parent_frame_id, "visual_odom");
        assert_eq!(parsed[0].child_frame_id, "base");
        assert_eq!(parsed[0].pose_variances, [0.0; 6]);
    }

    #[test]
    fn an_imu_from_an_unconfigured_frame_is_ignored() {
        let mut core = FusionCore::new(base_config(true));
        let mut stray = imu_message(0, 0.0, 0.0);
        stray.frame_id = "some_other_imu".to_string();
        for _ in 0..20 {
            core.handle_imu(&stray, &identity_mount());
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
                &imu_message(sample * NS_PER_SEC / 100, 0.0, 0.0),
                &identity_mount(),
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
            let ts_ns = sample * NS_PER_SEC / 100;
            core.handle_imu(&imu_message(ts_ns, 0.0, 0.0), &identity_mount());
            let mut second = imu_message(ts_ns + NS_PER_SEC / 200, 0.0, 0.0);
            second.frame_id = "imu2".to_string();
            second.angular_velocity = Vector3::new(0.02, 0.0, 0.0); // this gyro alone is biased
            core.handle_imu(&second, &identity_mount());
        }
        assert!(core.initialized);
        assert_eq!(core.filter.x.biases.len(), 2);
        assert!(core.filter.x.biases[0].gyro.norm() < 1e-9);
        assert!((core.filter.x.biases[1].gyro.x - 0.02).abs() < 1e-9);
    }

    #[test]
    fn a_nan_pose_is_dropped_rather_than_wedging_the_filter() {
        let mut config = base_config(false);
        config.odom_sources = one_source("visual_odom");
        let mut core = FusionCore::new(config);
        core.handle_source(&source_message("visual_odom", 0, 10.0));

        let mut nan_pose = source_message("visual_odom", NS_PER_SEC / 20, 10.25);
        nan_pose.pose.rotation =
            UnitQuaternion::from_quaternion(Quaternion::new(0.0, 0.0, 0.0, 0.0));
        assert!(nan_pose.pose.rotation.coords.iter().any(|v| v.is_nan()));
        core.handle_source(&nan_pose);

        core.handle_source(&source_message("visual_odom", NS_PER_SEC / 10, 10.5));
        let position = published_position(&mut core).expect("a period has elapsed");
        assert!((position.x - 0.5).abs() < 1e-12);
    }

    #[test]
    fn maybe_publish_waits_for_the_period_and_stamps_odom_to_base() {
        let mut config = base_config(false);
        config.publish_rate = 10.0;
        let mut core = FusionCore::new(config);
        core.handle_source(&source_message("odom", 0, 0.0));
        assert!(core.maybe_publish().is_none());
        core.handle_source(&source_message("odom", NS_PER_SEC / 20, 0.5));
        assert!(core.maybe_publish().is_none());
        core.handle_source(&source_message("odom", NS_PER_SEC / 10, 1.0));
        let msg = core.maybe_publish().expect("a full period has elapsed");
        assert_eq!(msg.frame_id, "odom");
        assert_eq!(msg.child_frame_id, "base");
    }

    #[test]
    fn the_outlier_gate_rejects_a_wild_pose_and_accepts_a_consistent_one() {
        let mut core = FusionCore::new(base_config(true));
        for sample in 0..5 {
            core.handle_imu(
                &imu_message(sample * NS_PER_SEC / 100, 0.0, 0.0),
                &identity_mount(),
            );
        }
        // Two to anchor: the first delta needs a predecessor, the second adopts the filter pose.
        core.handle_source(&source_message("odom", NS_PER_SEC / 20, 0.0));
        core.handle_source(&source_message("odom", NS_PER_SEC / 10, 0.0));
        core.handle_source(&source_message("odom", 3 * NS_PER_SEC / 20, 1000.0));
        let after_outlier = published_position(&mut core).expect("a period has elapsed");
        assert!(after_outlier.x.abs() < 1e-9);
        // From the outlier's own pose, so this delta is the small one.
        core.handle_source(&source_message("odom", NS_PER_SEC / 5, 1000.01));
        let after_inlier = published_position(&mut core).expect("a period has elapsed");
        assert!(after_inlier.x > 0.005);
    }

    #[test]
    fn a_state_past_the_position_cap_is_held_back_unless_the_cap_is_zero() {
        let mut config = base_config(false);
        config.max_position_m = 0.5;
        let mut core = FusionCore::new(config.clone());
        core.handle_source(&source_message("odom", 0, 0.0));
        core.handle_source(&source_message("odom", NS_PER_SEC / 10, 1.0));
        assert!(core.maybe_publish().is_none());

        config.max_position_m = 0.0;
        let mut core = FusionCore::new(config);
        core.handle_source(&source_message("odom", 0, 0.0));
        core.handle_source(&source_message("odom", NS_PER_SEC / 10, 1.0));
        assert!(core.maybe_publish().is_some());
    }

    #[test]
    fn the_same_imu_and_source_sequence_replays_bit_identically() {
        let mount = identity_mount();
        let run = || {
            let mut config = base_config(true);
            // Position only: the source reports no rotation, so it must not argue against the yaw.
            config.odom_sources[0].pose_variances = [1e-4, 1e-4, 1e-4, 0.0, 0.0, 0.0];
            let mut core = FusionCore::new(config);
            let mut published = Vec::new();
            for step in 0..60_i64 {
                let ts_ns = step * NS_PER_SEC / 100;
                let moving = step >= 10;
                let gyro_z = if moving { 0.3 } else { 0.0 };
                let accel_x = if moving { 0.5 } else { 0.0 };
                core.handle_imu(&imu_message(ts_ns, gyro_z, accel_x), &mount);
                if step % 10 == 0 {
                    core.handle_source(&source_message("odom", ts_ns, step as f64 * 0.02));
                }
                if let Some(msg) = core.maybe_publish() {
                    let position = msg.pose.translation.vector;
                    published.push((position.x, position.y, position.z, msg.pose.rotation.w));
                }
            }
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
