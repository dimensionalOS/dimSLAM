// Copyright 2026 Dimensional Inc.
// SPDX-License-Identifier: Apache-2.0
//
// The odometry filter: Point-LIO's estimator, run the way Point-LIO runs it so the two can be
// compared frame by frame, bit for bit, and grown around it to take any set of sensors.
//
// With a lidar, scans drive it as they drive Point-LIO: each scan (merged with any other
// lidar's scan that overlaps it) is thinned and cut into groups of points taken at the same
// instant; before each group the filter predicts up to every sample of the first IMU in
// between and takes that sample as a measurement, then predicts to the group and updates with
// its points; the scan's points then grow the map and the pose at the scan's end is published.
// Without a lidar the first IMU's samples drive it, and without either, the measurements do.
// Every other sensor (further IMUs, cameras, wheels, poses) is a measurement taken in time
// order between the first IMU's samples.

use std::collections::VecDeque;

use nalgebra::{Matrix3, UnitQuaternion, Vector3};

use super::lio_filter::{
    initial_covariance, process_noise, BiasAt, ImuMeasurement, ImuMount, Innovation,
    LidarMeasurement, LioFilter, LioState, PointGroup, StateMatrix, CHI_SQUARED_MEDIAN,
};
use super::scan_sync::LevelConfig;
use super::{DeskewConfig, DeskewFrame, InitialStds, LidarConfig, MapConfig};
use super::point_map::{to_world, PointMap};
use super::scan_points::{surface_points, voxel_centroids, SurfacePoint};
use super::scan_sync::{ros_seconds, Level, ScanSync};
use crate::types::LidarScan;

/// The filter's own settings; each lidar's and IMU's come with them. Point-LIO's defaults.
#[derive(Clone, Debug)]
pub struct LioConfig {
    pub map: MapConfig,
    /// The covariance the filter starts from.
    pub initial: InitialStds,
    /// How the first IMU levels the filter.
    pub level: LevelConfig,
    /// Random-walk variances of velocity, angular velocity, specific force and the biases.
    pub process: [f64; 5],
    /// See `LioFilter::outlier_gate`.
    pub outlier_gate: f64,
    /// Zero-twist [vx vy vz wx wy wz] fused after each source's measurement: >0 pins that
    /// axis to zero with this variance (dimSLAM main's `per_dimension_error_variance`).
    pub zero_twist: [f64; 6],
    pub deskew: DeskewConfig,
}

impl Default for LioConfig {
    fn default() -> Self {
        Self {
            map: MapConfig::default(),
            initial: InitialStds::default(),
            level: LevelConfig::default(),
            process: [20.0, 1000.0, 500.0, 0.0001, 0.0001],
            outlier_gate: -1.0,
            zero_twist: [0.0; 6],
            deskew: DeskewConfig::default(),
        }
    }
}

/// Point-LIO's IMU: its measurement noise and clip levels.
pub fn point_lio_imu() -> ImuMeasurement {
    ImuMeasurement {
        gyro_variance: 0.01,
        accel_variance: 0.01,
        clip: Some((35.0, 3.0)),
        clip_fraction: 0.99,
        accel_scale: LevelConfig::default().gravity / 1.0,
    }
}

/// One IMU as the filter takes it: how its readings are weighed, its bias random walks (gyro,
/// accel) and, when given as densities, the samples its rate is measured over first.
#[derive(Clone, Copy, Debug)]
pub struct ImuSetup {
    pub measurement: ImuMeasurement,
    pub walks: [f64; 2],
    /// [gyro, accel] noise densities and [gyro, accel] bias random walks; 0 keeps the value
    /// above. Converted at the IMU's measured rate.
    pub densities: [f64; 4],
    pub rate_samples: u32,
}

impl ImuSetup {
    pub fn new(measurement: ImuMeasurement, walks: [f64; 2]) -> Self {
        Self {
            measurement,
            walks,
            densities: [0.0; 4],
            rate_samples: 100,
        }
    }

    fn has_densities(&self) -> bool {
        self.densities.iter().any(|d| *d > 0.0)
    }
}

/// A scan waiting for the filter to pass its end: its points in its lidar's frame.
struct PendingDeskew {
    lidar: usize,
    begin: f64,
    end: f64,
    points: Vec<SurfacePoint>,
}

/// One input scan's points, each moved by the filter's pose at its own instant to where it
/// lies at the scan's end, in the configured frame.
#[derive(Clone, Debug)]
pub struct DeskewedScan {
    /// Index of the lidar it came from, in the configured order.
    pub lidar: usize,
    /// s: the scan's end.
    pub stamp: f64,
    pub points: Vec<SurfacePoint>,
}

/// The pose published for one scan.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct LioPose {
    /// s, the scan's end.
    pub stamp: f64,
    pub position: Vector3<f64>,
    /// [x, y, z, w].
    pub quaternion: [f64; 4],
    pub velocity: Vector3<f64>,
    pub angular_velocity: Vector3<f64>,
}

/// One IMU: how it is read, where it sits (once tf has said) and where its bias lives.
struct Imu {
    measurement: ImuMeasurement,
    setup: ImuSetup,
    /// Samples seen and the first one's stamp, while its densities wait for its rate.
    rate_seen: u32,
    rate_first: f64,
    /// Its noise is final: given as variances, or converted from densities at its rate.
    calibrated: bool,
    mount: Option<(Matrix3<f64>, Vector3<f64>)>,
    bias: BiasAt,
    /// Its previous sample's stamp and gyro reading: their difference is the angular
    /// acceleration its lever arm turns into linear acceleration.
    previous: Option<(f64, Vector3<f64>)>,
}

pub struct LioOdometry {
    config: LioConfig,
    pub sync: ScanSync,
    pub map: PointMap,
    pub filter: LioFilter,
    noise: StateMatrix,
    imus: Vec<Imu>,
    /// Per lidar, its mount (lidar to body) once tf has said.
    lidars: Vec<Option<(Matrix3<f64>, Vector3<f64>)>>,
    /// Per lidar, how its points are thinned and matched.
    lidar_configs: Vec<LidarConfig>,
    map_seed: Vec<SurfacePoint>,
    map_ready: bool,
    first_scan: bool,
    /// When the state, and the covariance, were last moved forward.
    state_at: f64,
    covariance_at: f64,
    /// The first IMU's sample the filter is between, and its stamp.
    rate: (Vector3<f64>, Vector3<f64>),
    rate_at: f64,
    /// Per point slot, across scans.
    pub world: Vec<SurfacePoint>,
    pub neighbors: Vec<Vec<SurfacePoint>>,
    pub used: Vec<bool>,
    /// The last scan's thinned points, each lidar's in its own frame at their own times.
    pub thinned: Vec<SurfacePoint>,
    thinned_lidar: Vec<usize>,
    scans_pushed: u64,
    late_points: u64,
    /// Motion measured by other sensors, oldest first.
    measured: VecDeque<Measured>,
    /// The state's angular velocity and body-frame velocity at the instants measurements were
    /// last checked for (s, rad/s, m/s), reaching back to the newest one taken: what a late
    /// measurement is measured against.
    motion_history: VecDeque<(f64, Vector3<f64>, Vector3<f64>)>,
    measured_pushed: bool,
    /// Per measuring source, every innovation over the chi-squared median its given variances
    /// predict, sorted: their median is how much noisier the source is than it was said to be.
    residuals: Vec<RunningMedian>,
    /// Per measuring source, measurements taken and refused by the gate.
    taken: Vec<[u64; 2]>,
    /// With deskew on: the filter's pose after each group of points it fused (s, world from
    /// body), reaching back to the oldest scan still waiting to be deskewed.
    trajectory: VecDeque<(f64, UnitQuaternion<f64>, Vector3<f64>)>,
    deskew_pending: VecDeque<PendingDeskew>,
    deskewed: VecDeque<DeskewedScan>,
    /// s: the longest span of a scan queued for deskewing.
    longest_scan: f64,
}

/// What a sensor other than the first IMU measured.
#[derive(Clone, Copy, Debug)]
enum Motion {
    /// The body's angular velocity (rad/s, body frame) and its per-axis variances.
    Rate(Vector3<f64>, Vector3<f64>),
    /// The body's velocity (m/s, body frame) and its per-axis variances.
    Velocity(Vector3<f64>, Vector3<f64>),
    /// The body's pose in the filter's world and [x y z roll pitch yaw] variances.
    Pose(Vector3<f64>, Matrix3<f64>, [f64; 6]),
    /// A further IMU's sample: rad/s and the accelerometer's unit.
    Imu(usize, Vector3<f64>, Vector3<f64>),
}

#[derive(Clone, Copy, Debug)]
struct Measured {
    stamp: f64,
    source: usize,
    motion: Motion,
}

impl LioOdometry {
    /// `imus` in order, the first driving the filter between scans (its random walks replace
    /// `config.process`'s); `lidars` in order.
    pub fn new(mut config: LioConfig, imus: Vec<ImuSetup>, lidars: Vec<LidarConfig>) -> Self {
        if let Some(first) = imus.first() {
            config.process[3] = first.walks[0];
            config.process[4] = first.walks[1];
        }
        let [v, w, a, bg, ba] = config.process;
        let map = PointMap::new(config.map.cube, config.map.capacity);
        let initial = initial_covariance(&config.initial);
        let mut filter = LioFilter {
            x: LioState::default(),
            p: initial,
            outlier_gate: config.outlier_gate,
            ..LioFilter::default()
        };
        let imus: Vec<Imu> = imus
            .into_iter()
            .enumerate()
            .map(|(i, setup)| {
                let [bg, ba] = setup.walks;
                let bias = if i == 0 {
                    BiasAt::Core
                } else {
                    let at = filter.extra.len();
                    let variances: Vec<f64> = (24..30).map(|c| initial[(c, c)]).collect();
                    filter.add_extra(&[0.0; 6], &variances, &[bg, bg, bg, ba, ba, ba]);
                    BiasAt::Extra(at)
                };
                Imu {
                    measurement: setup.measurement,
                    setup,
                    rate_seen: 0,
                    rate_first: 0.0,
                    calibrated: !setup.has_densities(),
                    mount: None,
                    bias,
                    previous: None,
                }
            })
            .collect();
        Self {
            sync: ScanSync::with_level(!imus.is_empty(), config.level),
            config,
            map,
            filter,
            noise: process_noise(v, w, a, bg, ba),
            imus,
            lidars: vec![None; lidars.len()],
            lidar_configs: lidars,
            map_seed: Vec::new(),
            map_ready: false,
            first_scan: true,
            state_at: 0.0,
            covariance_at: 0.0,
            rate: (Vector3::zeros(), Vector3::zeros()),
            rate_at: 0.0,
            world: Vec::new(),
            neighbors: Vec::new(),
            used: Vec::new(),
            thinned: Vec::new(),
            thinned_lidar: Vec::new(),
            scans_pushed: 0,
            late_points: 0,
            measured: VecDeque::new(),
            motion_history: VecDeque::new(),
            measured_pushed: false,
            residuals: Vec::new(),
            taken: Vec::new(),
            trajectory: VecDeque::new(),
            deskew_pending: VecDeque::new(),
            deskewed: VecDeque::new(),
            longest_scan: 0.0,
        }
        .with_lidar_waits()
    }

    fn with_lidar_waits(mut self) -> Self {
        for (lidar, config) in self.lidar_configs.iter().enumerate() {
            self.sync.set_max_wait_periods(lidar, config.max_wait_periods);
        }
        self
    }

    pub fn imu_mounted(&self, imu: usize) -> bool {
        self.imus.get(imu).is_some_and(|i| i.mount.is_some())
    }

    /// Where IMU `imu` sits: body_from_imu rotation and its origin in the body frame.
    pub fn set_imu_mount(&mut self, imu: usize, rotation: Matrix3<f64>, lever: Vector3<f64>) {
        if let Some(imu) = self.imus.get_mut(imu) {
            imu.mount = Some((rotation, lever));
        }
    }

    pub fn lidar_mounted(&self, lidar: usize) -> bool {
        self.lidars.get(lidar).is_some_and(Option::is_some)
    }

    /// Where lidar `lidar` sits: body_from_lidar rotation and its origin in the body frame.
    pub fn set_lidar_mount(
        &mut self,
        lidar: usize,
        rotation: Matrix3<f64>,
        translation: Vector3<f64>,
    ) {
        if let Some(mount) = self.lidars.get_mut(lidar) {
            *mount = Some((rotation, translation));
        }
    }

    /// `stamp` in seconds, `gyro` rad/s, `accel` in the accelerometer's unit; dropped until
    /// the IMU's mount is set.
    pub fn push_imu(&mut self, imu: usize, stamp: f64, gyro: Vector3<f64>, accel: Vector3<f64>) {
        if !self.imu_mounted(imu) {
            return;
        }
        if !self.calibrate(imu, stamp) && imu != 0 {
            // a further IMU waits for its rate before its densities can weigh it
            return;
        }
        if imu == 0 {
            self.sync.push_imu(stamp, gyro, accel);
        } else {
            self.push_measured(stamp, usize::MAX, Motion::Imu(imu, gyro, accel));
        }
    }

    /// Counts an IMU's samples until its rate is measured, then turns its densities into the
    /// per-reading variances and per-s^2 random walks the filter weighs it by. True once its
    /// noise is final.
    fn calibrate(&mut self, index: usize, stamp: f64) -> bool {
        let Some(imu) = self.imus.get_mut(index) else {
            return false;
        };
        if imu.calibrated {
            return true;
        }
        imu.rate_seen += 1;
        if imu.rate_seen == 1 {
            imu.rate_first = stamp;
        }
        let span = stamp - imu.rate_first;
        if imu.rate_seen < imu.setup.rate_samples.max(2) || span.is_nan() || span <= 0.0 {
            return false;
        }
        let rate = f64::from(imu.rate_seen - 1) / span;
        let [gyro_density, accel_density, gyro_walk, accel_walk] = imu.setup.densities;
        let converted = |density: f64, given: f64| {
            if density > 0.0 {
                density * density * rate
            } else {
                given
            }
        };
        imu.measurement.gyro_variance = converted(gyro_density, imu.measurement.gyro_variance);
        imu.measurement.accel_variance = converted(accel_density, imu.measurement.accel_variance);
        let walks = [
            converted(gyro_walk, imu.setup.walks[0]),
            converted(accel_walk, imu.setup.walks[1]),
        ];
        imu.calibrated = true;
        let bias = imu.bias;
        match bias {
            BiasAt::Core => {
                let [v, w, a, _, _] = self.config.process;
                self.config.process[3] = walks[0];
                self.config.process[4] = walks[1];
                self.noise = process_noise(v, w, a, walks[0], walks[1]);
            }
            BiasAt::Extra(at) => {
                for i in 0..6 {
                    if let Some(q) = self.filter.q_extra.get_mut(at + i) {
                        *q = walks[usize::from(i >= 3)];
                    }
                }
            }
        }
        tracing::info!(
            imu = index,
            rate_hz = rate,
            gyro_variance = self.imus.get(index).map_or(0.0, |i| i.measurement.gyro_variance),
            accel_variance = self.imus.get(index).map_or(0.0, |i| i.measurement.accel_variance),
            gyro_bias_random_walk = walks[0],
            accel_bias_random_walk = walks[1],
            "imu noise converted from its densities at its measured rate"
        );
        true
    }

    /// An angular velocity source `source` measured at `stamp` (s), body frame, with its
    /// per-axis variances (the floor of its learned spread; zero leaves an axis unmeasured).
    pub fn push_rate(
        &mut self,
        source: usize,
        stamp: f64,
        rate: Vector3<f64>,
        variances: Vector3<f64>,
    ) {
        self.push_measured(stamp, source, Motion::Rate(rate, variances));
    }

    /// The body's velocity source `source` measured at `stamp` (s), m/s in the body frame
    /// (legged or wheeled odometry, a camera's step), with per-axis variances as for a rate.
    pub fn push_body_velocity(
        &mut self,
        source: usize,
        stamp: f64,
        velocity: Vector3<f64>,
        variances: Vector3<f64>,
    ) {
        self.push_measured(stamp, source, Motion::Velocity(velocity, variances));
    }

    /// The body's pose in the filter's world source `source` measured at `stamp` (s), with
    /// [x y z roll pitch yaw] variances as for a rate.
    pub fn push_pose(
        &mut self,
        source: usize,
        stamp: f64,
        position: Vector3<f64>,
        rotation: Matrix3<f64>,
        variances: [f64; 6],
    ) {
        self.push_measured(stamp, source, Motion::Pose(position, rotation, variances));
    }

    /// The filter's pose and velocity now, in its world.
    pub fn state(&self) -> &LioState {
        &self.filter.x
    }

    pub fn started(&self) -> bool {
        !self.first_scan
    }

    /// Sensors interleave out of order; each one's own order is kept.
    fn push_measured(&mut self, stamp: f64, source: usize, motion: Motion) {
        let at = self.measured.partition_point(|m| m.stamp <= stamp);
        self.measured.insert(
            at,
            Measured {
                stamp,
                source,
                motion,
            },
        );
        self.measured_pushed = true;
    }

    /// The state's angular velocity and body velocity at `stamp`, from the history (its oldest
    /// entry when `stamp` is older still).
    fn motion_at(&self, stamp: f64) -> Option<(Vector3<f64>, Vector3<f64>)> {
        let after = self.motion_history.iter().position(|(t, _, _)| *t >= stamp);
        match after {
            Some(0) => self.motion_history.front().map(|(_, w, v)| (*w, *v)),
            Some(i) => {
                let (t0, w0, v0) = self.motion_history[i - 1];
                let (t1, w1, v1) = self.motion_history[i];
                let along = if t1 > t0 {
                    (stamp - t0) / (t1 - t0)
                } else {
                    1.0
                };
                Some((w0 + (w1 - w0) * along, v0 + (v1 - v0) * along))
            }
            None => self.motion_history.back().map(|(_, w, v)| (*w, *v)),
        }
    }

    fn body_velocity(&self) -> Vector3<f64> {
        self.filter.x.rotation.transpose() * self.filter.x.velocity
    }

    /// How much noisier than its given variances `source` has proven: the median of its
    /// innovations against the spread those variances predict, over what that median is when
    /// they are right (a median, so outliers don't set it), and never below one.
    pub fn noise_scale(&self, source: usize) -> f64 {
        self.scale(source)
    }

    fn scale(&self, source: usize) -> f64 {
        self.residuals
            .get(source)
            .and_then(RunningMedian::median)
            .map_or(1.0, |median| median.max(1.0))
    }

    /// Per measuring source: measurements taken, and refused by the gate.
    pub fn taken(&self) -> &[[u64; 2]] {
        &self.taken
    }

    fn learn(&mut self, source: usize, innovation: Option<Innovation>) {
        let Some(innovation) = innovation else { return };
        if self.taken.len() <= source {
            self.taken.resize(source + 1, [0, 0]);
        }
        self.taken[source][usize::from(!innovation.accepted)] += 1;
        if self.residuals.len() <= source {
            self.residuals
                .resize_with(source + 1, RunningMedian::default);
        }
        self.residuals[source]
            .insert(innovation.squared / CHI_SQUARED_MEDIAN[innovation.dof.min(6) - 1]);
    }

    /// Runs an update that observes the 30 states only, carrying it to any extra states.
    fn carrying<T>(&mut self, update: impl FnOnce(&mut LioFilter) -> T) -> T {
        if self.filter.extra.is_empty() {
            return update(&mut self.filter);
        }
        let (before, before_p) = (self.filter.x.clone(), self.filter.p);
        let out = update(&mut self.filter);
        self.filter.carry_extra(&before, &before_p);
        out
    }

    /// Moves the state, and the covariance, forward to `stamp`.
    fn predict_to(&mut self, stamp: f64) {
        self.filter
            .predict(stamp - self.state_at, &self.noise, true, false);
        self.state_at = stamp;
        let since = stamp - self.covariance_at;
        if since > 0.0 {
            self.covariance_at = stamp;
            self.filter.predict(since, &self.noise, false, true);
        }
    }

    /// Takes the measurements stamped before `until`, each at its own instant. One that arrives
    /// after the state has passed its stamp (a camera's frames trail the IMU) is taken at the
    /// state's instant, carried there by the state's own change since the stamp, so only the
    /// sensor's noise is left between the two.
    fn take_rates(&mut self, until: f64) {
        // Only a filter given measurements keeps the history; it is trimmed as they are taken.
        if !self.first_scan && self.measured_pushed {
            let body = self.body_velocity();
            self.motion_history
                .push_back((self.state_at, self.filter.x.angular_velocity, body));
        }
        while self.measured.front().is_some_and(|m| m.stamp < until) {
            let Some(Measured {
                stamp: measured_at,
                source,
                motion,
            }) = self.measured.pop_front() else {
                break;
            };
            if self.first_scan {
                continue;
            }
            let late = measured_at < self.state_at;
            let late_from = late.then(|| self.motion_at(measured_at)).flatten();
            while self
                .motion_history
                .get(1)
                .is_some_and(|(t, _, _)| *t <= measured_at)
            {
                self.motion_history.pop_front();
            }
            let stamp = measured_at.max(self.state_at);
            self.predict_to(stamp);
            self.measure(stamp, measured_at, source, motion, late_from);
        }
    }

    fn measure(
        &mut self,
        stamp: f64,
        measured_at: f64,
        source: usize,
        motion: Motion,
        late_from: Option<(Vector3<f64>, Vector3<f64>)>,
    ) {
        let from_source = !matches!(motion, Motion::Imu(..));
        self.measure_motion(stamp, measured_at, source, motion, late_from);
        if from_source && !self.imus.is_empty() {
            self.hold_zero_twist();
        }
    }

    /// The configured zero-twist axes, fused as body velocity and angular velocity of zero.
    fn hold_zero_twist(&mut self) {
        let [vx, vy, vz, wx, wy, wz] = self.config.zero_twist;
        let (linear, angular) = (Vector3::new(vx, vy, vz), Vector3::new(wx, wy, wz));
        if linear.iter().any(|v| *v > 0.0) {
            self.filter
                .update_body_velocity(&Vector3::zeros(), &linear, 1.0);
        }
        if angular.iter().any(|v| *v > 0.0) {
            self.filter
                .update_angular_velocity(&Vector3::zeros(), &angular, 1.0);
        }
    }

    fn measure_motion(
        &mut self,
        stamp: f64,
        measured_at: f64,
        source: usize,
        motion: Motion,
        late_from: Option<(Vector3<f64>, Vector3<f64>)>,
    ) {
        match motion {
            Motion::Rate(mut rate, variances) => {
                if let Some((then, _)) = late_from {
                    rate += self.filter.x.angular_velocity - then;
                }
                let scale = self.scale(source);
                let innovation = self
                    .filter
                    .update_angular_velocity(&rate, &variances, scale);
                self.learn(source, innovation);
            }
            Motion::Velocity(mut velocity, variances) => {
                if let Some((_, then)) = late_from {
                    velocity += self.body_velocity() - then;
                }
                let scale = self.scale(source);
                let innovation = self
                    .filter
                    .update_body_velocity(&velocity, &variances, scale);
                self.learn(source, innovation);
            }
            Motion::Pose(mut position, rotation, variances) => {
                // Carried over the time the state has moved on since it was measured.
                position += self.filter.x.velocity * (stamp - measured_at);
                let scale = self.scale(source);
                let innovation = self
                    .filter
                    .update_pose(&position, &rotation, &variances, scale);
                self.learn(source, innovation);
            }
            Motion::Imu(index, gyro, accel) => {
                let Some(imu) = self.imus.get_mut(index) else {
                    return;
                };
                let Some((rotation, lever)) = imu.mount else {
                    return;
                };
                let measurement = imu.measurement;
                let mount = angular_motion(
                    rotation,
                    lever,
                    imu.previous,
                    (measured_at, gyro),
                    measurement.gyro_variance,
                );
                imu.previous = Some((measured_at, gyro));
                let bias = imu.bias;
                self.filter
                    .update_mounted_imu(&gyro, &accel, &measurement, &mount, bias);
            }
        }
    }

    /// Scans handed in so far, and points that arrived after the filter had passed them.
    pub fn scans_pushed(&self) -> u64 {
        self.scans_pushed
    }

    pub fn late_points(&self) -> u64 {
        self.late_points
    }

    /// Scans and IMU samples waiting for the filter to take them.
    pub fn pending(&self) -> (usize, usize) {
        (self.sync.pending_scans(), self.sync.imus.len())
    }

    /// Points the map holds; at rest it stops growing.
    pub fn map_points(&self) -> usize {
        self.map.point_count()
    }

    /// Dropped until the lidar's mount is set.
    pub fn push_scan(&mut self, lidar: usize, scan: &LidarScan) {
        if !self.lidar_mounted(lidar) {
            return;
        }
        self.scans_pushed += 1;
        let Some(config) = self.lidar_configs.get(lidar) else {
            return;
        };
        let points = surface_points(scan, config.min_range, config.keep_every, config.duplicate_tolerance);
        if self.config.deskew.enabled && self.map_ready {
            let all = surface_points(scan, config.min_range, 1, config.duplicate_tolerance);
            let begin = ros_seconds(scan.timestamp_ns as u64 as f64 / 1e9);
            let end = begin + all.iter().map(|p| p.time_ms).fold(0.0f32, f32::max) as f64 / 1000.0;
            self.longest_scan = self.longest_scan.max(end - begin);
            self.deskew_pending.push_back(PendingDeskew {
                lidar,
                begin,
                end,
                points: all,
            });
        }
        self.sync.push_scan(lidar, scan.timestamp_ns as u64, points);
    }

    /// Runs the filter on as far as its inputs reach: `None` when they reach no further, else
    /// whether it published a pose.
    pub fn step(&mut self) -> Option<Option<LioPose>> {
        if !self.lidars.is_empty() {
            return self.step_scan();
        }
        if !self.imus.is_empty() {
            return self.step_imu();
        }
        self.step_measured()
    }

    fn pose(&self, stamp: f64) -> LioPose {
        let x = &self.filter.x;
        LioPose {
            stamp,
            position: x.position,
            quaternion: quaternion(&x.rotation),
            velocity: x.velocity,
            angular_velocity: x.angular_velocity,
        }
    }

    /// The level the first IMU found, turned from its axes to the body's.
    fn level_body(&mut self, level: Level) {
        let mut x = LioState {
            rotation: level.rotation,
            gravity: level.gravity,
            specific_force: level.specific_force,
            ..LioState::default()
        };
        let Some((rotation, _)) = self.imus.first().and_then(|imu| imu.mount) else {
            return;
        };
        if rotation != Matrix3::identity() {
            x.rotation = level.rotation * rotation.transpose();
            x.specific_force = rotation * level.specific_force;
        }
        self.filter.x = x;
    }

    /// No lidar: the first IMU's samples move the filter.
    fn step_imu(&mut self) -> Option<Option<LioPose>> {
        if let Some(level) = self.sync.level_without_scans() {
            self.level_body(level);
        }
        if self.sync.imus.is_empty() {
            return None;
        }
        let now = self.sync.newest_imu();
        if !self.first_scan && now <= self.state_at {
            return None;
        }
        self.catch_up_imu(now);
        Some(Some(self.pose(ros_seconds(now))))
    }

    /// Neither lidar nor IMU: each measurement moves the filter to its own stamp.
    fn step_measured(&mut self) -> Option<Option<LioPose>> {
        let newest = self.measured.back()?.stamp;
        if self.first_scan {
            self.first_scan = false;
            self.state_at = newest.min(self.measured.front().map_or(newest, |m| m.stamp));
            self.covariance_at = self.state_at;
        }
        self.take_rates(f64::INFINITY);
        Some(Some(self.pose(ros_seconds(newest.max(self.state_at)))))
    }

    /// Runs the next scan whose IMU samples have all arrived.
    fn step_scan(&mut self) -> Option<Option<LioPose>> {
        let scan = self.sync.next_scan()?;
        if let Some(level) = scan.level {
            self.level_body(level);
        }
        if scan.parts.iter().all(|(_, points)| points.is_empty()) {
            return Some(None);
        }
        self.thin(scan.parts);
        let n = self.thinned.len();
        if !self.map_ready {
            let x = &self.filter.x;
            for (p, &lidar) in self.thinned.iter().zip(&self.thinned_lidar) {
                let Some(Some((mount_rotation, mount_translation))) = self.lidars.get(lidar).copied()
                else {
                    continue;
                };
                self.map_seed.push(to_world(
                    p,
                    &x.rotation,
                    &x.position,
                    &mount_rotation,
                    &mount_translation,
                ));
            }
            if self.map_seed.len() < self.config.map.seed_points {
                return Some(None);
            }
            self.map.insert(&self.map_seed);
            self.map_seed = Vec::new();
            self.map_ready = true;
            self.sync.set_map_ready();
            self.filter.anchor_pose();
            return Some(None);
        }
        let blank = SurfacePoint {
            position: [0.0; 3],
            reflectivity: 0.0,
            time_ms: 0.0,
        };
        self.world.resize(n, blank);
        self.neighbors.resize(n, Vec::new());
        if self.used.len() < n {
            self.used.resize(n, true);
        }
        let mut start = 0usize;
        // Points the filter had already passed: never matched, so never grown into the map either.
        let mut late = Vec::new();
        for run in self.runs() {
            let (Some(last), Some(&lidar)) = (
                self.thinned.get(start + run - 1),
                self.thinned_lidar.get(start),
            ) else {
                break;
            };
            let now = last.time_ms as f64 / 1000.0 + scan.begin;
            if !self.first_scan && now < self.state_at {
                // Another lidar's points the filter has already passed.
                self.late_points += run as u64;
                late.push(start..start + run);
                start += run;
                continue;
            }
            self.catch_up_imu(now);
            let (Some(Some((mount_rotation, mount_translation))), Some(config)) =
                (self.lidars.get(lidar).copied(), self.lidar_configs.get(lidar))
            else {
                start += run;
                continue;
            };
            let m = LidarMeasurement {
                mount_rotation,
                mount_translation,
                point_variance: config.point_variance,
                plane_tolerance: config.plane_tolerance,
                range_per_distance_sq: config.range_per_distance_sq,
                neighbor_range: config.neighbor_range,
                plane_points: config.plane_points,
            };
            let mut group = PointGroup {
                body: &self.thinned[start..start + run],
                world: &mut self.world[start..start + run],
                neighbors: &mut self.neighbors[start..start + run],
                used: &mut self.used[start..start + run],
            };
            if n >= 1 {
                let map = &mut self.map;
                if self.filter.extra.is_empty() {
                    self.filter.update_points(&mut group, map, &m);
                } else {
                    let (before, before_p) = (self.filter.x.clone(), self.filter.p);
                    self.filter.update_points(&mut group, map, &m);
                    self.filter.carry_extra(&before, &before_p);
                }
            }
            start += run;
            self.record_trajectory(now);
        }
        let pose = self.pose(ros_seconds(scan.end));
        self.deskew_ready();
        if n > self.config.map.grow_min_points {
            if late.is_empty() {
                self.map.grow(
                    &self.world[..n],
                    &self.neighbors[..n],
                    self.config.map.spacing,
                );
            } else {
                // A skipped point's world position and neighbors are a previous scan's, left in the buffers.
                let mut fused = vec![true; n];
                for range in late {
                    fused[range].fill(false);
                }
                let (world, neighbors): (Vec<_>, Vec<_>) = (0..n)
                    .filter(|&i| fused[i])
                    .map(|i| (self.world[i], std::mem::take(&mut self.neighbors[i])))
                    .unzip();
                self.map.grow(&world, &neighbors, self.config.map.spacing);
            }
        }
        Some(Some(pose))
    }

    /// Each lidar's points thinned in its own frame, then all of them in time order (a tie
    /// keeps the lidars' order).
    fn thin(&mut self, parts: Vec<(usize, Vec<SurfacePoint>)>) {
        let mut merged: Vec<(SurfacePoint, usize)> = Vec::new();
        for (lidar, points) in parts {
            let cube = self.lidar_configs.get(lidar).map_or(0.0, |c| c.scan_cube);
            merged.extend(
                voxel_centroids(&points, cube)
                    .into_iter()
                    .map(|p| (p, lidar)),
            );
        }
        if merged.windows(2).any(|pair| pair[0].1 != pair[1].1) {
            merged.sort_by(|a, b| a.0.time_ms.total_cmp(&b.0.time_ms));
        }
        self.thinned = merged.iter().map(|(p, _)| *p).collect();
        self.thinned_lidar = merged.iter().map(|(_, l)| *l).collect();
    }

    /// Keeps the filter's pose at `now` for deskewing, dropping what no waiting scan needs.
    fn record_trajectory(&mut self, now: f64) {
        if !self.config.deskew.enabled {
            return;
        }
        let x = &self.filter.x;
        self.trajectory
            .push_back((now, UnitQuaternion::from_matrix(&x.rotation), x.position));
        // A scan still in flight, as long as the longest seen, may reach back before them all.
        let oldest = self
            .deskew_pending
            .iter()
            .map(|scan| scan.begin)
            .fold(now, f64::min)
            - self.longest_scan;
        while self.trajectory.get(1).is_some_and(|(t, _, _)| *t <= oldest) {
            self.trajectory.pop_front();
        }
    }

    /// The filter's pose at `t` (world from body), between the two recorded around it.
    fn pose_at(&self, t: f64) -> Option<(UnitQuaternion<f64>, Vector3<f64>)> {
        let after = self.trajectory.partition_point(|(at, _, _)| *at < t);
        let (t1, q1, p1) = self.trajectory.get(after).or(self.trajectory.back())?;
        let Some((t0, q0, p0)) = after.checked_sub(1).and_then(|i| self.trajectory.get(i)) else {
            return Some((*q1, *p1));
        };
        if t1 <= t0 || t >= *t1 {
            return Some((*q1, *p1));
        }
        let along = (t - t0) / (t1 - t0);
        Some((q0.slerp(q1, along), p0 + (p1 - p0) * along))
    }

    /// Deskews each waiting scan the filter has passed the end of; one too old for the
    /// trajectory (taken before the filter started) is dropped, and a point before its first
    /// pose takes that pose.
    fn deskew_ready(&mut self) {
        let (Some(&(first, _, _)), Some(&(newest, _, _))) =
            (self.trajectory.front(), self.trajectory.back())
        else {
            return;
        };
        let mut waiting = VecDeque::with_capacity(self.deskew_pending.len());
        while let Some(scan) = self.deskew_pending.pop_front() {
            if scan.end > newest {
                waiting.push_back(scan);
                continue;
            }
            if scan.end < first {
                continue;
            }
            if let Some(deskewed) = self.deskew(scan) {
                self.deskewed.push_back(deskewed);
            }
        }
        self.deskew_pending = waiting;
    }

    fn deskew(&self, scan: PendingDeskew) -> Option<DeskewedScan> {
        let (mount_rotation, mount_translation) = (*self.lidars.get(scan.lidar)?)?;
        let (end_rotation, end_position) = self.pose_at(scan.end)?;
        let to_end = end_rotation.inverse();
        let frame = self.config.deskew.frame;
        let mut points = Vec::with_capacity(scan.points.len());
        for p in &scan.points {
            let Some((rotation, position)) = self.pose_at(scan.begin + p.time_ms as f64 / 1000.0)
            else {
                continue;
            };
            let lidar = Vector3::new(
                p.position[0] as f64,
                p.position[1] as f64,
                p.position[2] as f64,
            );
            let world = rotation * (mount_rotation * lidar + mount_translation) + position;
            let out = match frame {
                DeskewFrame::Odom => world,
                DeskewFrame::Output => to_end * (world - end_position),
            };
            points.push(SurfacePoint {
                position: [out.x as f32, out.y as f32, out.z as f32],
                ..*p
            });
        }
        let voxel = self.config.deskew.voxel;
        if voxel > 0.0 {
            points = voxel_centroids(&points, voxel);
        }
        Some(DeskewedScan {
            lidar: scan.lidar,
            stamp: scan.end,
            points,
        })
    }

    /// Scans deskewed since the last call, oldest first.
    pub fn take_deskewed(&mut self) -> impl Iterator<Item = DeskewedScan> + '_ {
        self.deskewed.drain(..)
    }

    /// Lengths of the runs of thinned points taken at one instant by one lidar.
    fn runs(&self) -> Vec<usize> {
        let mut runs = Vec::with_capacity(self.thinned.len());
        let mut length = 0;
        for i in 1..self.thinned.len() {
            length += 1;
            if self.thinned[i].time_ms > self.thinned[i - 1].time_ms
                || self.thinned_lidar[i] != self.thinned_lidar[i - 1]
            {
                runs.push(length);
                length = 0;
            }
        }
        runs.push(length + 1);
        runs
    }

    /// Takes every IMU sample before `now` (predicting up to it, then measuring with it) and
    /// predicts the state on to `now`.
    fn catch_up_imu(&mut self, now: f64) {
        let sync = &mut self.sync;
        if self.first_scan {
            while sync.inertial && now > sync.imu_after.stamp {
                let Some(next) = sync.imus.pop_front() else {
                    break;
                };
                sync.imu_before = sync.imu_after;
                sync.imu_after = next;
            }
            self.rate = (sync.imu_before.gyro, sync.imu_before.accel_g);
            self.rate_at = sync.imu_before.stamp;
            self.first_scan = false;
            self.covariance_at = now;
            self.state_at = now;
        }
        if !self.sync.imus.is_empty() {
            while now > self.sync.imu_after.stamp {
                let sample = self.sync.imu_after;
                self.take_rates(sample.stamp);
                let previous = (self.rate_at, self.rate.0);
                self.rate = (sample.gyro, sample.accel_g);
                self.rate_at = sample.stamp;
                self.filter
                    .predict(sample.stamp - self.state_at, &self.noise, true, false);
                self.state_at = sample.stamp;
                let since = sample.stamp - self.covariance_at;
                if since > 0.0 {
                    self.covariance_at = sample.stamp;
                    self.filter.predict(since, &self.noise, false, true);
                    self.update_first_imu(previous);
                }
                let sync = &mut self.sync;
                let Some(next) = sync.imus.pop_front() else {
                    break;
                };
                sync.imu_before = sync.imu_after;
                sync.imu_after = next;
            }
        }
        self.take_rates(now);
        self.filter
            .predict(now - self.state_at, &self.noise, true, false);
        self.state_at = now;
        if !self.sync.inertial {
            // With no IMU to pace it, the covariance grows at every step.
            self.predict_to(now);
        }
    }

    /// The first IMU's sample in `self.rate`, after the one at `previous` (stamp, gyro).
    fn update_first_imu(&mut self, previous: (f64, Vector3<f64>)) {
        let Some(imu) = self.imus.first() else {
            return;
        };
        let Some((rotation, lever)) = imu.mount else {
            return;
        };
        let measurement = imu.measurement;
        let (gyro, accel) = self.rate;
        if rotation == Matrix3::identity() && lever == Vector3::zeros() {
            return self.carrying(|filter| filter.update_imu(&gyro, &accel, &measurement));
        }
        let mount = angular_motion(
            rotation,
            lever,
            Some(previous),
            (self.rate_at, gyro),
            measurement.gyro_variance,
        );
        self.carrying(|filter| {
            filter.update_mounted_imu(&gyro, &accel, &measurement, &mount, BiasAt::Core)
        });
    }
}

/// An IMU's mount with the angular acceleration between its `previous` sample and this one
/// (stamp, gyro): a difference of two gyro readings, so its variance is twice one reading's
/// over the interval squared.
fn angular_motion(
    rotation: Matrix3<f64>,
    lever: Vector3<f64>,
    previous: Option<(f64, Vector3<f64>)>,
    (at, gyro): (f64, Vector3<f64>),
    gyro_variance: f64,
) -> ImuMount {
    let (angular_acceleration, angular_acceleration_variance) = match previous {
        Some((t0, gyro0)) if at > t0 => {
            let dt = at - t0;
            (
                rotation * (gyro - gyro0) / dt,
                2.0 * gyro_variance / (dt * dt),
            )
        }
        _ => (Vector3::zeros(), 0.0),
    };
    ImuMount {
        rotation,
        lever,
        angular_acceleration,
        angular_acceleration_variance,
    }
}

/// The running median of non-negative values, in two heaps: O(log n) per value.
#[derive(Debug, Default)]
struct RunningMedian {
    /// The lower half (a max-heap) and the upper half (a min-heap), as the values' bits,
    /// which order as the values do when they are non-negative.
    lower: std::collections::BinaryHeap<u64>,
    upper: std::collections::BinaryHeap<std::cmp::Reverse<u64>>,
}

impl RunningMedian {
    fn insert(&mut self, value: f64) {
        let bits = value.max(0.0).to_bits();
        if self.lower.peek().is_some_and(|top| bits > *top) {
            self.upper.push(std::cmp::Reverse(bits));
        } else {
            self.lower.push(bits);
        }
        if self.lower.len() > self.upper.len() + 1 {
            if let Some(moved) = self.lower.pop() {
                self.upper.push(std::cmp::Reverse(moved));
            }
        } else if self.upper.len() > self.lower.len() {
            if let Some(std::cmp::Reverse(moved)) = self.upper.pop() {
                self.lower.push(moved);
            }
        }
    }

    fn median(&self) -> Option<f64> {
        self.lower.peek().map(|bits| f64::from_bits(*bits))
    }
}

/// The unit quaternion [x, y, z, w] of a rotation matrix (Shoemake's method).
pub fn quaternion(m: &Matrix3<f64>) -> [f64; 4] {
    let mut q = [0.0; 4];
    let mut t = m[(0, 0)] + (m[(1, 1)] + m[(2, 2)]);
    if t > 0.0 {
        t = (t + 1.0).sqrt();
        q[3] = 0.5 * t;
        t = 0.5 / t;
        q[0] = (m[(2, 1)] - m[(1, 2)]) * t;
        q[1] = (m[(0, 2)] - m[(2, 0)]) * t;
        q[2] = (m[(1, 0)] - m[(0, 1)]) * t;
    } else {
        let mut i = 0;
        if m[(1, 1)] > m[(0, 0)] {
            i = 1;
        }
        if m[(2, 2)] > m[(i, i)] {
            i = 2;
        }
        let j = (i + 1) % 3;
        let k = (j + 1) % 3;
        t = (m[(i, i)] - m[(j, j)] - m[(k, k)] + 1.0).sqrt();
        q[i] = 0.5 * t;
        t = 0.5 / t;
        q[3] = (m[(k, j)] - m[(j, k)]) * t;
        q[j] = (m[(j, i)] + m[(i, j)]) * t;
        q[k] = (m[(k, i)] + m[(i, k)]) * t;
    }
    q
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::odometry_fusion::lio_filter::exp;

    /// A lidar on a body yawing at 2 rad/s while its scan sweeps a fixed ring of points: read in
    /// the turning frame they smear, deskewed they land back where the scan's end sees them.
    #[test]
    fn deskew_undoes_a_rotating_scan() {
        let w = 2.0;
        let lever = Vector3::new(0.3, 0.0, 0.1);
        let yaw = |t: f64| UnitQuaternion::from_scaled_axis(Vector3::new(0.0, 0.0, w * t));
        let moved = |t: f64| Vector3::new(0.5 * t, 0.0, 0.0);
        for frame in [DeskewFrame::Output, DeskewFrame::Odom] {
            let mut config = LioConfig::default();
            config.deskew.enabled = true;
            config.deskew.frame = frame;
            let mut lio = LioOdometry::new(config, Vec::new(), vec![LidarConfig::default()]);
            lio.set_lidar_mount(0, Matrix3::identity(), lever);
            let (begin, end) = (10.0, 10.1);
            for i in 0..=20 {
                let t = begin + 0.005 * i as f64;
                lio.trajectory.push_back((t, yaw(t), moved(t)));
            }
            let world: Vec<Vector3<f64>> = (0..100)
                .map(|i| {
                    let a = i as f64 * 0.0628;
                    Vector3::new(5.0 * a.cos(), 5.0 * a.sin(), 0.5)
                })
                .collect();
            let points = world
                .iter()
                .enumerate()
                .map(|(i, p)| {
                    let t = begin + 0.001 * i as f64;
                    let lidar = yaw(t).inverse() * (p - moved(t)) - lever;
                    SurfacePoint {
                        position: [lidar.x as f32, lidar.y as f32, lidar.z as f32],
                        reflectivity: 0.0,
                        time_ms: (1.0 * i as f64) as f32,
                    }
                })
                .collect();
            lio.deskew_pending.push_back(PendingDeskew {
                lidar: 0,
                begin,
                end,
                points,
            });
            lio.deskew_ready();
            let scans: Vec<DeskewedScan> = lio.take_deskewed().collect();
            assert_eq!(scans.len(), 1);
            assert_eq!(scans[0].stamp, end);
            for (p, w) in scans[0].points.iter().zip(&world) {
                let expected = match frame {
                    DeskewFrame::Odom => *w,
                    DeskewFrame::Output => yaw(end).inverse() * (w - moved(end)),
                };
                let got = Vector3::new(p.position[0] as f64, p.position[1] as f64, p.position[2] as f64);
                assert!((got - expected).norm() < 1e-4, "{frame:?}: {got} vs {expected}");
            }
        }
    }

    /// A body at the origin, still for a second (to level), then spun up over half a second to
    /// turn about z at `w` rad/s, read by IMUs at `mounts` (body_from_imu rotation, lever,
    /// gyro bias), 200 Hz.
    fn turning(
        mounts: &[(Matrix3<f64>, Vector3<f64>, Vector3<f64>)],
        w: f64,
        seconds: f64,
    ) -> LioOdometry {
        let imus = mounts
            .iter()
            .map(|_| ImuSetup::new(point_lio_imu(), [0.0001, 0.0001]))
            .collect();
        let mut lio = LioOdometry::new(LioConfig::default(), imus, Vec::new());
        for (i, (rotation, lever, _)) in mounts.iter().enumerate() {
            lio.set_imu_mount(i, *rotation, *lever);
        }
        let up = Vector3::new(0.0, 0.0, 9.81);
        for k in 0..(seconds * 200.0) as usize {
            let t = k as f64 / 200.0;
            let spin_up = ((t - 1.0) / 0.5).clamp(0.0, 1.0);
            let rate = Vector3::new(0.0, 0.0, w * spin_up);
            let acceleration = Vector3::new(
                0.0,
                0.0,
                if spin_up > 0.0 && spin_up < 1.0 {
                    w / 0.5
                } else {
                    0.0
                },
            );
            for (i, (rotation, lever, bias)) in mounts.iter().enumerate() {
                let specific_force =
                    up + rate.cross(&rate.cross(lever)) + acceleration.cross(lever);
                let gyro = rotation.transpose() * rate + bias;
                let accel = rotation.transpose() * specific_force / 9.81;
                lio.push_imu(i, t, gyro, accel);
            }
            while lio.step().is_some() {}
        }
        lio
    }

    #[test]
    fn an_imu_away_from_the_body_reads_its_turn_and_its_lever_arm() {
        // Tilted on every axis and 0.3 m out: turning at 2 rad/s it feels 1.2 m/s^2 inward.
        let mount = (
            exp(&Vector3::new(-0.88, 0.88, -1.38)),
            Vector3::new(0.3, 0.0, 0.0),
            Vector3::zeros(),
        );
        let lio = turning(&[mount], 2.0, 4.0);
        let x = lio.state();
        assert!(
            (x.angular_velocity - Vector3::new(0.0, 0.0, 2.0)).norm() < 0.05,
            "{}",
            x.angular_velocity
        );
        assert!(
            x.velocity.norm() < 0.05,
            "the centripetal pull read as motion: {}",
            x.velocity
        );
        assert!(x.position.norm() < 0.05, "{}", x.position);
    }

    #[test]
    fn a_second_imu_learns_how_its_gyro_bias_differs() {
        let body = (Matrix3::identity(), Vector3::zeros(), Vector3::zeros());
        let bias = Vector3::new(0.05, -0.03, 0.02);
        let second = (
            exp(&Vector3::new(0.0, 0.0, std::f64::consts::FRAC_PI_2)),
            Vector3::new(0.0, 0.2, 0.0),
            bias,
        );
        let lio = turning(&[body, second], 1.0, 10.0);
        // Two gyros tell only their biases' difference apart (nothing says which one is
        // off), so that is what the filter must have found, in the body's axes.
        let learned = lio.filter.extra.fixed_rows::<3>(0).into_owned();
        let difference = second.0 * learned - lio.state().gyro_bias;
        assert!(
            (difference - second.0 * bias).norm() < 0.005,
            "{difference}"
        );
        // Nor which one to believe, so the turn is off by no more than the bias between them.
        let turn_error = (lio.state().angular_velocity - Vector3::new(0.0, 0.0, 1.0)).norm();
        assert!(turn_error < bias.norm(), "{turn_error}");
    }

    /// One 100 ms sweep of a 10 x 10 x 4 m box room seen from `origin` (body frame = lidar frame),
    /// points on a `spacing` grid of each wall, timed by azimuth.
    fn room_sweep(origin: Vector3<f64>, spacing: f64) -> Vec<([f32; 3], u32)> {
        let mut points = Vec::new();
        let steps = |lo: f64, hi: f64| {
            (0..=((hi - lo) / spacing).round() as i32).map(move |i| lo + spacing * i as f64)
        };
        for a in steps(-5.0, 5.0) {
            for z in steps(-1.0, 3.0) {
                for wall in [
                    Vector3::new(5.0, a, z),
                    Vector3::new(-5.0, a, z),
                    Vector3::new(a, 5.0, z),
                    Vector3::new(a, -5.0, z),
                ] {
                    let p = wall - origin;
                    let azimuth = p.y.atan2(p.x) + std::f64::consts::PI;
                    let offset = (azimuth / (2.0 * std::f64::consts::PI) * 99e6) as u32;
                    points.push(([p.x as f32, p.y as f32, p.z as f32], offset));
                }
            }
        }
        points.sort_by_key(|(_, t)| *t);
        points
    }

    #[test]
    fn late_points_never_grow_the_map_so_a_still_robot_costs_the_same_forever() {
        // A second, denser lidar whose stream runs 0.4 s behind: the filter never waits for it,
        // so all its points are late. They used to be grown into the map from whatever the
        // per-point buffers held (a previous scan's points, or blanks at the origin), every scan.
        let mut lio = LioOdometry::new(
            LioConfig::default(),
            vec![ImuSetup::new(point_lio_imu(), [0.0001, 0.0001])],
            vec![LidarConfig::default(), LidarConfig::default()],
        );
        lio.set_imu_mount(0, Matrix3::identity(), Vector3::zeros());
        for lidar in 0..2 {
            lio.set_lidar_mount(lidar, Matrix3::identity(), Vector3::zeros());
        }
        let origin = Vector3::new(0.3, -0.2, 0.0);
        let sweeps = [room_sweep(origin, 0.5), room_sweep(origin, 0.2)];
        let scan = |lidar: usize, start_s: f64| LidarScan {
            timestamp_ns: (start_s * 1e9).round() as i64,
            frame_id: format!("lidar{lidar}"),
            points: sweeps[lidar].iter().map(|(p, _)| *p).collect(),
            offsets_ns: sweeps[lidar].iter().map(|(_, t)| *t).collect(),
            ..Default::default()
        };
        let mut imu_t = 0.0;
        let mut sizes = Vec::new();
        for k in 0..300usize {
            let begin = 1.0 + k as f64 * 0.1;
            while imu_t <= begin + 0.15 {
                lio.push_imu(0, imu_t, Vector3::zeros(), Vector3::new(0.0, 0.0, 1.0));
                imu_t += 0.005;
            }
            lio.push_scan(0, &scan(0, begin));
            if k >= 4 {
                lio.push_scan(1, &scan(1, begin - 0.4 + 0.005));
            }
            while lio.step().is_some() {}
            sizes.push(lio.map_points());
        }
        assert!(lio.late_points() > 0, "the test must make late points");
        let (early, late) = (sizes[150], sizes[299]);
        assert!(late <= early + early / 50, "the map grew from {early} to {late} points at rest");
        assert!(lio.state().position.norm() < 0.05, "{}", lio.state().position);
    }

    #[test]
    fn the_running_median_is_the_middle_value() {
        let mut median = RunningMedian::default();
        for (value, expected) in [(5.0, 5.0), (1.0, 1.0), (3.0, 3.0), (9.0, 3.0), (7.0, 5.0)] {
            median.insert(value);
            assert_eq!(median.median(), Some(expected));
        }
    }

    #[test]
    fn quaternion_matches_the_rotation_on_both_branches() {
        for v in [Vector3::new(0.0, 0.0, 1.0), Vector3::new(3.0, 0.1, 0.0)] {
            let r = exp(&v);
            let [x, y, z, w] = quaternion(&r);
            let back =
                nalgebra::UnitQuaternion::from_quaternion(nalgebra::Quaternion::new(w, x, y, z));
            assert!((back.to_rotation_matrix().into_inner() - r).norm() < 1e-12);
        }
    }
}
