// Copyright 2026 Dimensional Inc.
// SPDX-License-Identifier: Apache-2.0
//
// Error-state Kalman filter fusing IMU propagation with nav_msgs/Odometry sources told apart
// by header.frame_id. Without an IMU there is no process model: see `blend`.

mod eskf;

use std::collections::{HashMap, VecDeque};

use dimos_module::nalgebra::{DVector, Isometry3, Translation3, UnitQuaternion, Vector3};
use dimos_module::{native_config, warn_throttled, Transform};
use lcm_msgs::nav_msgs::Odometry;
use lcm_msgs::sensor_msgs::Imu;
use tracing::{info, warn};

use eskf::{Filter, Jacobian, Mat15, State, Vec15};

const NS_PER_SEC: i64 = 1_000_000_000;

/// Bias calibration restarts above this rate: above the gyro's own bias, below real rotation.
const STATIONARY_GYRO_LIMIT: f64 = 0.05;

/// Raw gyro differences are too noisy for the tangential lever-arm term. A time constant, not
/// a per-sample weight, so the cutoff does not move with IMU rate.
const ANGULAR_ACCEL_TIME_CONSTANT: f64 = 0.025;

fn stamp_to_ns(header: &lcm_msgs::std_msgs::Header) -> i64 {
    header.stamp.sec as i64 * NS_PER_SEC + header.stamp.nsec as i64
}

fn to_stamp(timestamp_ns: i64) -> lcm_msgs::std_msgs::Time {
    lcm_msgs::std_msgs::Time {
        sec: (timestamp_ns / NS_PER_SEC) as i32,
        nsec: (timestamp_ns % NS_PER_SEC) as i32,
    }
}

fn vector3(v: &lcm_msgs::geometry_msgs::Vector3) -> Vector3<f64> {
    Vector3::new(v.x, v.y, v.z)
}

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

#[native_config]
#[derive(Clone)]
pub struct OdometryFusionConfig {
    pub odom_frame: String,
    pub base_frame: String,
    /// Output cadence; the filter itself runs at the IMU rate.
    pub publish_rate: f64,
    /// How far back a late measurement can reach before it is dropped.
    pub replay_buffer_seconds: f64,
    /// Outlier gate in standard deviations per measurement dimension; 0 disables.
    pub mahalanobis_gate: f64,
    /// Gate on the filter's own state, which can run away with no bad measurement; 0 disables.
    pub max_position_m: f64,
    /// Off bypasses the Kalman machinery and seeds level from the first source message; see `blend`.
    pub use_imu: bool,
    /// Required when use_imu is on; 0 counts as unset.
    pub imu_gyro_noise_density: f64,
    pub imu_gyro_random_walk: f64,
    pub imu_accel_noise_density: f64,
    pub imu_accel_random_walk: f64,
    pub gravity: f64,
    /// Samples averaged while stationary to level the filter and take the IMU biases.
    pub imu_init_samples: i64,
    pub initial_position_std: f64,
    pub initial_velocity_std: f64,
    pub initial_rotation_std: f64,
    pub initial_bias_std: f64,
    /// One entry per source: the header.frame_id its messages carry.
    pub source_frames: Vec<String>,
    /// 6 per source, [x y z roll pitch yaw]: <0 message covariance, 0 drop, >0 fixed.
    /// A drifting source's covariance describes accumulated drift, not the delta: prefer a fixed value.
    pub source_pose_variances: Vec<f64>,
    /// 6 per source, [vx vy vz wx wy wz], same convention, body frame.
    pub source_twist_variances: Vec<f64>,
    /// Virtual zero-twist [vx vy vz wx wy wz] fused after each source update (use_imu only):
    /// >0 pins that dimension to zero with this variance (e.g. non-holonomic vy, vz), 0 frees it.
    pub constraint_twist_variances: Vec<f64>,
}

#[derive(Clone, Debug)]
struct Measurement {
    source: usize,
    absolute: bool,
    pose: Option<Isometry3<f64>>,
    /// Consecutive-message delta when the source drifts.
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
    covariance: Mat15,
    anchors: Vec<Option<Isometry3<f64>>>,
    activity: Vec<SourceActivity>,
    last_imu_ns: i64,
    last_gyro: Vector3<f64>,
}

pub struct FusionCore {
    config: OdometryFusionConfig,
    filter: Filter,
    initialized: bool,
    init_gyro_sum: Vector3<f64>,
    init_accel_sum: Vector3<f64>,
    init_samples: u64,
    events: VecDeque<Event>,
    anchors: Vec<Option<Isometry3<f64>>>,
    activity: Vec<SourceActivity>,
    last_imu_ns: i64,
    last_gyro: Vector3<f64>,
    lever_gyro: Vector3<f64>,
    lever_gyro_ns: i64,
    lever_alpha: Vector3<f64>,
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
        Self {
            config,
            filter: Filter::default(),
            initialized: false,
            init_gyro_sum: Vector3::zeros(),
            init_accel_sum: Vector3::zeros(),
            init_samples: 0,
            events: VecDeque::new(),
            anchors: Vec::new(),
            activity: Vec::new(),
            last_imu_ns: 0,
            last_gyro: Vector3::zeros(),
            lever_gyro: Vector3::zeros(),
            lever_gyro_ns: 0,
            lever_alpha: Vector3::zeros(),
            previous_source_pose: HashMap::new(),
            last_publish_ns: 0,
            imu_samples: 0,
            measurements: 0,
            gated: 0,
            replayed: 0,
            too_late: 0,
        }
    }

    /// The filter's body frame is base_frame, so both IMU vectors are rotated through the mount.
    pub fn handle_imu(&mut self, imu: &Imu, base_from_imu: &Transform) {
        let lever = base_from_imu.translation();
        let base_from_imu = base_from_imu.rotation();
        let gyro = base_from_imu * vector3(&imu.angular_velocity);
        let mut accel = base_from_imu * vector3(&imu.linear_acceleration);
        let ts_ns = stamp_to_ns(&imu.header);
        if self.lever_gyro_ns != 0 {
            let dt = (ts_ns - self.lever_gyro_ns) as f64 / NS_PER_SEC as f64;
            if dt > 0.0 {
                let raw_alpha = (gyro - self.lever_gyro) / dt;
                let blend = 1.0 - (-dt / ANGULAR_ACCEL_TIME_CONSTANT).exp();
                self.lever_alpha += (raw_alpha - self.lever_alpha) * blend;
            }
        }
        self.lever_gyro = gyro;
        self.lever_gyro_ns = ts_ns;
        // The lever arm adds centripetal and tangential terms: mount kinematics, not base motion.
        accel -= gyro.cross(&gyro.cross(&lever)) + self.lever_alpha.cross(&lever);
        if !self.initialized {
            self.check_config();
            if gyro.norm() > STATIONARY_GYRO_LIMIT {
                warn_throttled!(
                    std::time::Duration::from_secs(5),
                    gyro_norm = gyro.norm(),
                    "imu init deferred: waiting for the robot to hold still",
                );
                self.init_gyro_sum = Vector3::zeros();
                self.init_accel_sum = Vector3::zeros();
                self.init_samples = 0;
                return;
            }
            self.init_gyro_sum += gyro;
            self.init_accel_sum += accel;
            self.init_samples += 1;
            if self.init_samples >= self.config.imu_init_samples.max(1) as u64 {
                let mean_accel = self.init_accel_sum / self.init_samples as f64;
                let mean_gyro = self.init_gyro_sum / self.init_samples as f64;
                // Stationary accel reads the specific force R^T (0,0,g): level so it maps to +z.
                let world_from_body = UnitQuaternion::rotation_between(&mean_accel, &Vector3::z())
                    .unwrap_or_default();
                // Any stationary reading beyond g is accel bias; left at zero it dead-reckons z away.
                let accel_bias = mean_accel
                    - world_from_body.inverse_transform_vector(&Vector3::new(
                        0.0,
                        0.0,
                        self.config.gravity,
                    ));
                self.initialize(ts_ns, world_from_body, mean_gyro, accel_bias);
            }
            return;
        }
        self.insert(Event {
            ts_ns,
            kind: EventKind::Imu { gyro, accel },
            state: State::default(),
            covariance: Mat15::identity(),
            anchors: Vec::new(),
            activity: Vec::new(),
            last_imu_ns: 0,
            last_gyro: Vector3::zeros(),
        });
        self.imu_samples += 1;
    }

    pub fn handle_source(&mut self, msg: &Odometry) {
        if !self.initialized {
            if self.config.use_imu {
                return;
            }
            self.check_config();
            self.initialize(
                stamp_to_ns(&msg.header),
                UnitQuaternion::identity(),
                Vector3::zeros(),
                Vector3::zeros(),
            );
        }
        let Some(source) = self
            .config
            .source_frames
            .iter()
            .position(|frame| *frame == msg.header.frame_id)
        else {
            warn!(frame_id = %msg.header.frame_id, "odometry from unconfigured frame_id dropped");
            return;
        };

        let absolute = msg.header.frame_id == self.config.odom_frame;
        let mut pose_variance = [0.0; 6];
        let mut twist_variance = [0.0; 6];
        for dim in 0..6 {
            pose_variance[dim] = resolve_variance(
                self.config.source_pose_variances[source * 6 + dim],
                msg.pose.covariance[dim * 7],
            );
            twist_variance[dim] = resolve_variance(
                self.config.source_twist_variances[source * 6 + dim],
                msg.twist.covariance[dim * 7],
            );
        }

        let orientation = UnitQuaternion::from_quaternion(dimos_module::nalgebra::Quaternion::new(
            msg.pose.pose.orientation.w,
            msg.pose.pose.orientation.x,
            msg.pose.pose.orientation.y,
            msg.pose.pose.orientation.z,
        ));
        let raw_norm = (msg.pose.pose.orientation.w.powi(2)
            + msg.pose.pose.orientation.x.powi(2)
            + msg.pose.pose.orientation.y.powi(2)
            + msg.pose.pose.orientation.z.powi(2))
        .sqrt();
        let pose_valid = (raw_norm - 1.0).abs() < 0.1;
        let pose_wanted = pose_variance.iter().any(|v| *v > 0.0);

        let mut measurement = Measurement {
            source,
            absolute,
            pose: None,
            delta: None,
            pose_variance,
            twist_variance,
            linear: Vector3::new(
                msg.twist.twist.linear.x,
                msg.twist.twist.linear.y,
                msg.twist.twist.linear.z,
            ),
            angular: Vector3::new(
                msg.twist.twist.angular.x,
                msg.twist.twist.angular.y,
                msg.twist.twist.angular.z,
            ),
        };
        if pose_wanted && pose_valid {
            let pose = Isometry3::from_parts(
                Translation3::new(
                    msg.pose.pose.position.x,
                    msg.pose.pose.position.y,
                    msg.pose.pose.position.z,
                ),
                orientation,
            );
            if absolute {
                measurement.pose = Some(pose);
            } else {
                if let Some(previous) = self.previous_source_pose.get(&source) {
                    measurement.delta = Some(previous.inverse() * pose);
                }
                self.previous_source_pose.insert(source, pose);
            }
        }

        self.insert(Event {
            ts_ns: stamp_to_ns(&msg.header),
            kind: EventKind::Measurement(measurement),
            state: State::default(),
            covariance: Mat15::identity(),
            anchors: Vec::new(),
            activity: Vec::new(),
            last_imu_ns: 0,
            last_gyro: Vector3::zeros(),
        });
        self.measurements += 1;
    }

    fn check_config(&self) {
        let sources = self.config.source_frames.len();
        assert!(
            self.config.source_pose_variances.len() == 6 * sources
                && self.config.source_twist_variances.len() == 6 * sources,
            "source_pose_variances and source_twist_variances need 6 entries per source_frames entry"
        );
        assert!(
            self.config.constraint_twist_variances.len() == 6,
            "constraint_twist_variances needs exactly 6 entries"
        );
        assert!(
            !self.config.use_imu
                || (self.config.imu_gyro_noise_density > 0.0
                    && self.config.imu_gyro_random_walk > 0.0
                    && self.config.imu_accel_noise_density > 0.0
                    && self.config.imu_accel_random_walk > 0.0),
            "use_imu needs all four imu noise figures set"
        );
    }

    fn initialize(
        &mut self,
        timestamp_ns: i64,
        world_from_body: UnitQuaternion<f64>,
        gyro_bias: Vector3<f64>,
        accel_bias: Vector3<f64>,
    ) {
        self.filter.noise = eskf::Noise {
            gyro_noise_density: self.config.imu_gyro_noise_density,
            gyro_random_walk: self.config.imu_gyro_random_walk,
            accel_noise_density: self.config.imu_accel_noise_density,
            accel_random_walk: self.config.imu_accel_random_walk,
        };
        self.filter.gravity = self.config.gravity;
        self.filter.init(
            world_from_body,
            gyro_bias,
            accel_bias,
            self.config.initial_position_std,
            self.config.initial_velocity_std,
            self.config.initial_rotation_std,
            self.config.initial_bias_std,
        );
        self.anchors = vec![None; self.config.source_frames.len()];
        self.activity = vec![SourceActivity::default(); self.config.source_frames.len()];
        self.initialized = true;
        self.last_publish_ns = timestamp_ns;
        let mut seed = Event {
            ts_ns: timestamp_ns,
            kind: EventKind::Seed,
            state: State::default(),
            covariance: Mat15::identity(),
            anchors: Vec::new(),
            activity: Vec::new(),
            last_imu_ns: 0,
            last_gyro: Vector3::zeros(),
        };
        self.snapshot(&mut seed);
        self.events.push_back(seed);
        info!(
            gyro_bias = gyro_bias.norm(),
            use_imu = self.config.use_imu,
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
                "measurement older than the replay buffer dropped",
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
            self.filter.p_cov = previous.covariance;
            self.anchors = previous.anchors.clone();
            self.activity = previous.activity.clone();
            self.last_imu_ns = previous.last_imu_ns;
            self.last_gyro = previous.last_gyro;
        }
        let ts_ns = self.events[index].ts_ns;
        let kind = self.events[index].kind.clone();
        match kind {
            EventKind::Seed => {}
            EventKind::Imu { gyro, accel } => {
                let dt = (ts_ns - self.last_imu_ns) as f64 / 1.0e9;
                if dt > 0.0 && dt < 1.0 {
                    self.filter.propagate(dt, &gyro, &accel);
                }
                self.last_imu_ns = ts_ns;
                self.last_gyro = gyro;
            }
            EventKind::Measurement(measurement) => {
                if self.config.use_imu {
                    self.apply(&measurement);
                } else {
                    self.blend(&measurement, ts_ns);
                }
            }
        }
        let mut event = std::mem::replace(
            &mut self.events[index],
            Event {
                ts_ns,
                kind: EventKind::Seed,
                state: State::default(),
                covariance: Mat15::identity(),
                anchors: Vec::new(),
                activity: Vec::new(),
                last_imu_ns: 0,
                last_gyro: Vector3::zeros(),
            },
        );
        self.snapshot(&mut event);
        self.events[index] = event;
    }

    fn apply(&mut self, measurement: &Measurement) {
        let mut rows: Vec<Vec15> = Vec::new();
        let mut residuals: Vec<f64> = Vec::new();
        let mut variances: Vec<f64> = Vec::new();

        let target = if measurement.absolute {
            measurement.pose
        } else if let Some(delta) = measurement.delta {
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
                    let mut row = Vec15::zeros();
                    row[dim] = 1.0;
                    rows.push(row);
                    residuals.push(position_residual[dim]);
                    variances.push(measurement.pose_variance[dim]);
                }
            }
            for dim in 0..3 {
                if measurement.pose_variance[3 + dim] > 0.0 {
                    let mut row = Vec15::zeros();
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
        if accepted && !measurement.absolute && target.is_some() {
            // Anchoring to the corrected pose, not each source's own chain, keeps drifters bounded.
            self.anchor(measurement.source);
        }

        let mut constraint_variance = [0.0; 6];
        constraint_variance.copy_from_slice(&self.config.constraint_twist_variances);
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

        let increment = if measurement.absolute {
            let Some(pose) = measurement.pose else { return };
            Isometry3::from_parts(Translation3::from(self.filter.x.position), self.filter.x.q).inverse()
                * pose
        } else {
            let Some(delta) = measurement.delta else { return };
            delta
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

    /// Body-frame twist rows. Angular rate is not a state: it is predicted from the last gyro sample.
    fn add_twist_rows(
        &self,
        linear: &Vector3<f64>,
        angular: &Vector3<f64>,
        variance: &[f64; 6],
        rows: &mut Vec<Vec15>,
        residuals: &mut Vec<f64>,
        variances: &mut Vec<f64>,
    ) {
        let body_from_world = self.filter.x.q.inverse().to_rotation_matrix();
        let body_velocity = body_from_world * self.filter.x.velocity;
        let velocity_skew = eskf::skew(&body_velocity);
        for dim in 0..3 {
            if variance[dim] > 0.0 {
                let mut row = Vec15::zeros();
                for col in 0..3 {
                    row[3 + col] = body_from_world.matrix()[(dim, col)];
                    row[6 + col] = velocity_skew[(dim, col)];
                }
                rows.push(row);
                residuals.push(linear[dim] - body_velocity[dim]);
                variances.push(variance[dim]);
            }
        }
        let predicted_angular = self.last_gyro - self.filter.x.gyro_bias;
        for dim in 0..3 {
            if variance[3 + dim] > 0.0 {
                let mut row = Vec15::zeros();
                row[9 + dim] = -1.0;
                rows.push(row);
                residuals.push(angular[dim] - predicted_angular[dim]);
                variances.push(variance[3 + dim]);
            }
        }
    }

    fn update(&mut self, rows: &[Vec15], residuals: &[f64], variances: &[f64]) -> bool {
        if rows.is_empty() {
            return false;
        }
        let residual = DVector::from_row_slice(residuals);
        let variance = DVector::from_row_slice(variances);
        let mut jacobian = Jacobian::zeros(rows.len());
        for (index, row) in rows.iter().enumerate() {
            jacobian.row_mut(index).copy_from(&row.transpose());
        }
        let accepted = self.filter.update(
            &residual,
            &jacobian,
            &variance,
            self.config.mahalanobis_gate,
        );
        if !accepted {
            self.gated += 1;
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
        event.covariance = self.filter.p_cov;
        event.anchors = self.anchors.clone();
        event.activity = self.activity.clone();
        event.last_imu_ns = self.last_imu_ns;
        event.last_gyro = self.last_gyro;
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
                "filter state is not publishable; holding the last pose",
            );
        }
        finite && capped
    }

    pub fn maybe_publish(&mut self) -> Option<(Odometry, Transform)> {
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
        let covariance = latest.covariance;
        let last_gyro = latest.last_gyro;
        let ts_ns = latest.ts_ns;

        let body_from_world = state.q.inverse().to_rotation_matrix();
        let body_velocity = body_from_world * state.velocity;
        let body_angular = last_gyro - state.gyro_bias;

        let mut msg = Odometry::default();
        msg.header.stamp = to_stamp(ts_ns);
        msg.header.frame_id = self.config.odom_frame.clone();
        msg.child_frame_id = self.config.base_frame.clone();
        msg.pose.pose.position.x = state.position.x;
        msg.pose.pose.position.y = state.position.y;
        msg.pose.pose.position.z = state.position.z;
        msg.pose.pose.orientation.x = state.q.i;
        msg.pose.pose.orientation.y = state.q.j;
        msg.pose.pose.orientation.z = state.q.k;
        msg.pose.pose.orientation.w = state.q.w;
        msg.twist.twist.linear.x = body_velocity.x;
        msg.twist.twist.linear.y = body_velocity.y;
        msg.twist.twist.linear.z = body_velocity.z;
        msg.twist.twist.angular.x = body_angular.x;
        msg.twist.twist.angular.y = body_angular.y;
        msg.twist.twist.angular.z = body_angular.z;
        let velocity_world = covariance.fixed_view::<3, 3>(3, 3).into_owned();
        let velocity_body =
            body_from_world.matrix() * velocity_world * body_from_world.matrix().transpose();
        for row in 0..3 {
            for col in 0..3 {
                msg.pose.covariance[row * 6 + col] = covariance[(row, col)];
                msg.pose.covariance[row * 6 + col + 3] = covariance[(row, col + 6)];
                msg.pose.covariance[(row + 3) * 6 + col] = covariance[(row + 6, col)];
                msg.pose.covariance[(row + 3) * 6 + col + 3] = covariance[(row + 6, col + 6)];
                msg.twist.covariance[row * 6 + col] = velocity_body[(row, col)];
                msg.twist.covariance[(row + 3) * 6 + col + 3] = covariance[(row + 9, col + 9)];
            }
        }

        let pose = Isometry3::from_parts(Translation3::from(state.position), state.q);
        let transform = Transform::new(
            self.config.odom_frame.clone(),
            self.config.base_frame.clone(),
            ts_ns as f64 / 1.0e9,
            pose,
        );
        Some((msg, transform))
    }

    pub fn report(&self) {
        info!(
            imu_samples = self.imu_samples,
            measurements = self.measurements,
            gated = self.gated,
            replayed = self.replayed,
            too_late = self.too_late,
            "odometry_fusion shutting down"
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const GRAVITY: f64 = 9.80665;

    fn base_config(use_imu: bool) -> OdometryFusionConfig {
        OdometryFusionConfig {
            odom_frame: "odom".to_string(),
            base_frame: "base".to_string(),
            publish_rate: 1000.0,
            replay_buffer_seconds: 2.0,
            mahalanobis_gate: 3.0,
            max_position_m: 10000.0,
            use_imu,
            imu_gyro_noise_density: 0.01,
            imu_gyro_random_walk: 0.001,
            imu_accel_noise_density: 0.1,
            imu_accel_random_walk: 0.01,
            gravity: GRAVITY,
            imu_init_samples: 5,
            initial_position_std: 0.1,
            initial_velocity_std: 0.1,
            initial_rotation_std: 0.05,
            initial_bias_std: 0.01,
            source_frames: vec!["odom".to_string()],
            source_pose_variances: vec![1e-4; 6],
            source_twist_variances: vec![0.0; 6],
            constraint_twist_variances: vec![0.0; 6],
        }
    }

    fn source_message(frame_id: &str, ts_ns: i64, x: f64) -> Odometry {
        let mut msg = Odometry::default();
        msg.header.stamp = to_stamp(ts_ns);
        msg.header.frame_id = frame_id.to_string();
        msg.pose.pose.position.x = x;
        msg.pose.pose.orientation.w = 1.0;
        msg
    }

    fn imu_message(ts_ns: i64, gyro_z: f64, accel_x: f64) -> Imu {
        let mut imu = Imu::default();
        imu.header.stamp = to_stamp(ts_ns);
        imu.header.frame_id = "imu".to_string();
        imu.angular_velocity.z = gyro_z;
        imu.linear_acceleration.x = accel_x;
        imu.linear_acceleration.z = GRAVITY;
        imu
    }

    fn identity_mount() -> Transform {
        Transform::new("base", "imu", 0.0, Isometry3::identity())
    }

    fn published_position(core: &mut FusionCore) -> Option<Vector3<f64>> {
        core.maybe_publish().map(|(msg, _)| {
            Vector3::new(
                msg.pose.pose.position.x,
                msg.pose.pose.position.y,
                msg.pose.pose.position.z,
            )
        })
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
        config.source_frames = vec!["visual_odom".to_string()];
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
    fn maybe_publish_waits_for_the_period_and_stamps_odom_to_base() {
        let mut config = base_config(false);
        config.publish_rate = 10.0;
        let mut core = FusionCore::new(config);
        core.handle_source(&source_message("odom", 0, 0.0));
        assert!(core.maybe_publish().is_none());
        core.handle_source(&source_message("odom", NS_PER_SEC / 20, 0.5));
        assert!(core.maybe_publish().is_none());
        core.handle_source(&source_message("odom", NS_PER_SEC / 10, 1.0));
        let (msg, transform) = core.maybe_publish().expect("a full period has elapsed");
        assert_eq!(msg.header.frame_id, "odom");
        assert_eq!(msg.child_frame_id, "base");
        assert_eq!(transform.parent, "odom");
        assert_eq!(transform.child, "base");
    }

    #[test]
    fn the_mahalanobis_gate_rejects_a_wild_pose_and_accepts_a_consistent_one() {
        let mut core = FusionCore::new(base_config(true));
        for sample in 0..5 {
            core.handle_imu(&imu_message(sample * NS_PER_SEC / 100, 0.0, 0.0), &identity_mount());
        }
        core.handle_source(&source_message("odom", NS_PER_SEC / 10, 1000.0));
        let after_outlier = published_position(&mut core).expect("a period has elapsed");
        assert!(after_outlier.x.abs() < 1e-9);
        core.handle_source(&source_message("odom", NS_PER_SEC / 5, 0.01));
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
            let mut core = FusionCore::new(base_config(true));
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
                if let Some((msg, _)) = core.maybe_publish() {
                    published.push((
                        msg.pose.pose.position.x,
                        msg.pose.pose.position.y,
                        msg.pose.pose.position.z,
                        msg.pose.pose.orientation.w,
                    ));
                }
            }
            published
        };
        let first = run();
        assert!(first.len() > 10);
        // Identical output proves nothing unless the filter actually moved and turned.
        let last = first.last().expect("published above");
        assert!(last.0.hypot(last.1) > 0.05, "the fused pose barely moved: {last:?}");
        assert!(last.3 < 0.999, "the fused pose barely turned: {last:?}");
        assert_eq!(first, run());
    }
}
