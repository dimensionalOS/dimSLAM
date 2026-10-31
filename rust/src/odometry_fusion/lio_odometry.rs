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

use nalgebra::{Matrix3, Vector3};

use super::scan_points::{surface_points, voxel_centroids, SurfacePoint};
use super::lio_filter::{
    initial_covariance, process_noise, BiasAt, ImuMeasurement, ImuMount, LidarMeasurement,
    LioFilter, LioState, PointGroup, StateMatrix,
};
use super::point_map::{to_world, PointMap};
use super::scan_sync::{ros_seconds, Level, ScanSync};
use crate::types::LidarScan;

/// Point-LIO's defaults, named for what they do.
#[derive(Clone, Debug)]
pub struct LioConfig {
    /// m: nearer points are the robot itself.
    pub min_range: f64,
    /// Keep every this many usable points of a scan.
    pub keep_every: u32,
    /// m: cube a scan is thinned to one point per.
    pub scan_cube: f32,
    /// m: cube of the map's coverage test.
    pub map_spacing: f64,
    /// m: cube of the map's hash.
    pub map_cube: f32,
    pub map_capacity: usize,
    /// Points the map needs before it is used.
    pub map_seed_points: usize,
    /// How points are matched and weighed; its mount is each lidar's own, from tf.
    pub lidar: LidarMeasurement,
    /// Random-walk variances of velocity, angular velocity, specific force and the biases.
    pub process: [f64; 5],
}

impl Default for LioConfig {
    fn default() -> Self {
        Self {
            min_range: 0.5,
            keep_every: 3,
            scan_cube: 0.2,
            map_spacing: 0.5,
            map_cube: 2.0,
            map_capacity: 1_000_000,
            map_seed_points: 10,
            lidar: LidarMeasurement {
                plane_tolerance: 0.1,
                range_per_distance_sq: 81.0,
                point_variance: 0.01,
                neighbor_range: 5.0,
                mount_rotation: Matrix3::identity(),
                mount_translation: Vector3::zeros(),
            },
            process: [20.0, 1000.0, 500.0, 0.0001, 0.0001],
        }
    }
}

/// Point-LIO's IMU: its measurement noise and clip levels.
pub fn point_lio_imu() -> ImuMeasurement {
    ImuMeasurement {
        gyro_variance: 0.01,
        accel_variance: 0.01,
        clip: Some((35.0, 3.0)),
        accel_scale: 9.81 / 1.0,
    }
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
    map_seed: Vec<SurfacePoint>,
    map_ready: bool,
    first_scan: bool,
    /// When the state, and the covariance, were last moved forward.
    state_at: f64,
    covariance_at: f64,
    /// The first IMU's sample the filter is between.
    rate: (Vector3<f64>, Vector3<f64>),
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
    /// Per measuring source, every residual taken against the state over the spread its given
    /// variances allow, sorted: their median is how much noisier the source is than it was
    /// said to be.
    residuals: Vec<Vec<f64>>,
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
    /// `imus` in order with their gyro and accel bias random walks, the first driving the
    /// filter between scans (its random walks replace `config.process`'s); `lidars` how many.
    pub fn new(mut config: LioConfig, imus: Vec<(ImuMeasurement, [f64; 2])>, lidars: usize) -> Self {
        if let Some((_, walks)) = imus.first() {
            config.process[3] = walks[0];
            config.process[4] = walks[1];
        }
        let [v, w, a, bg, ba] = config.process;
        let map = PointMap::new(config.map_cube, config.map_capacity);
        let mut filter = LioFilter {
            x: LioState::default(),
            p: initial_covariance(),
            ..LioFilter::default()
        };
        let initial = initial_covariance();
        let imus: Vec<Imu> = imus
            .into_iter()
            .enumerate()
            .map(|(i, (measurement, [bg, ba]))| {
                let bias = if i == 0 {
                    BiasAt::Core
                } else {
                    let at = filter.extra.len();
                    let variances: Vec<f64> = (24..30).map(|c| initial[(c, c)]).collect();
                    filter.add_extra(&[0.0; 6], &variances, &[bg, bg, bg, ba, ba, ba]);
                    BiasAt::Extra(at)
                };
                Imu { measurement, mount: None, bias, previous: None }
            })
            .collect();
        Self {
            sync: ScanSync::new(!imus.is_empty()),
            config,
            map,
            filter,
            noise: process_noise(v, w, a, bg, ba),
            imus,
            lidars: vec![None; lidars],
            map_seed: Vec::new(),
            map_ready: false,
            first_scan: true,
            state_at: 0.0,
            covariance_at: 0.0,
            rate: (Vector3::zeros(), Vector3::zeros()),
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
        }
    }

    pub fn imu_mounted(&self, imu: usize) -> bool {
        self.imus.get(imu).is_some_and(|i| i.mount.is_some())
    }

    /// Where IMU `imu` sits: body_from_imu rotation and its origin in the body frame.
    pub fn set_imu_mount(&mut self, imu: usize, rotation: Matrix3<f64>, lever: Vector3<f64>) {
        self.imus[imu].mount = Some((rotation, lever));
    }

    pub fn lidar_mounted(&self, lidar: usize) -> bool {
        self.lidars.get(lidar).is_some_and(Option::is_some)
    }

    /// Where lidar `lidar` sits: body_from_lidar rotation and its origin in the body frame.
    pub fn set_lidar_mount(&mut self, lidar: usize, rotation: Matrix3<f64>, translation: Vector3<f64>) {
        self.lidars[lidar] = Some((rotation, translation));
    }

    /// `stamp` in seconds, `gyro` rad/s, `accel` in the accelerometer's unit; dropped until
    /// the IMU's mount is set.
    pub fn push_imu(&mut self, imu: usize, stamp: f64, gyro: Vector3<f64>, accel: Vector3<f64>) {
        if !self.imu_mounted(imu) {
            return;
        }
        if imu == 0 {
            self.sync.push_imu(stamp, gyro, accel);
        } else {
            self.push_measured(stamp, usize::MAX, Motion::Imu(imu, gyro, accel));
        }
    }

    /// An angular velocity source `source` measured at `stamp` (s), body frame, with its
    /// per-axis variances (the floor of its learned spread; zero leaves an axis unmeasured).
    pub fn push_rate(&mut self, source: usize, stamp: f64, rate: Vector3<f64>, variances: Vector3<f64>) {
        self.push_measured(stamp, source, Motion::Rate(rate, variances));
    }

    /// The body's velocity source `source` measured at `stamp` (s), m/s in the body frame
    /// (legged or wheeled odometry, a camera's step), with per-axis variances as for a rate.
    pub fn push_body_velocity(&mut self, source: usize, stamp: f64, velocity: Vector3<f64>, variances: Vector3<f64>) {
        self.push_measured(stamp, source, Motion::Velocity(velocity, variances));
    }

    /// The body's pose in the filter's world source `source` measured at `stamp` (s), with
    /// [x y z roll pitch yaw] variances as for a rate.
    pub fn push_pose(&mut self, source: usize, stamp: f64, position: Vector3<f64>, rotation: Matrix3<f64>, variances: [f64; 6]) {
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
        self.measured.insert(at, Measured { stamp, source, motion });
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
                let along = if t1 > t0 { (stamp - t0) / (t1 - t0) } else { 1.0 };
                Some((w0 + (w1 - w0) * along, v0 + (v1 - v0) * along))
            }
            None => self.motion_history.back().map(|(_, w, v)| (*w, *v)),
        }
    }

    fn body_velocity(&self) -> Vector3<f64> {
        self.filter.x.rotation.transpose() * self.filter.x.velocity
    }

    /// Variances scaled by how much noisier than them `source` has proven: its residuals'
    /// median absolute value (x1.4826 for a normal's sigma), so outliers don't set it, and
    /// never below the given ones.
    fn learned<const N: usize>(&mut self, source: usize, residual: &[f64; N], variances: &[f64; N]) -> [f64; N] {
        let measured: Vec<usize> = (0..N).filter(|&a| variances[a] > 0.0).collect();
        if measured.is_empty() {
            return *variances;
        }
        let normalized = (measured.iter().map(|&a| residual[a] * residual[a] / variances[a]).sum::<f64>()
            / measured.len() as f64)
            .sqrt();
        if self.residuals.len() <= source {
            self.residuals.resize_with(source + 1, Vec::new);
        }
        let median = insert_sorted_median(&mut self.residuals[source], normalized);
        let scale = (1.4826 * median).powi(2).max(1.0);
        variances.map(|v| v * scale)
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
            let Measured { stamp: measured_at, source, motion } =
                self.measured.pop_front().expect("checked");
            if self.first_scan {
                continue;
            }
            let late = measured_at < self.state_at;
            let late_from = late.then(|| self.motion_at(measured_at)).flatten();
            while self.motion_history.get(1).is_some_and(|(t, _, _)| *t <= measured_at) {
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
        match motion {
            Motion::Rate(mut rate, variances) => {
                if let Some((then, _)) = late_from {
                    rate += self.filter.x.angular_velocity - then;
                }
                let residual = rate - self.filter.x.angular_velocity;
                let variances = Vector3::from(self.learned(source, &residual.into(), &variances.into()));
                self.filter.update_angular_velocity(&rate, &variances);
            }
            Motion::Velocity(mut velocity, variances) => {
                let body = self.body_velocity();
                if let Some((_, then)) = late_from {
                    velocity += body - then;
                }
                let residual = velocity - body;
                let variances = Vector3::from(self.learned(source, &residual.into(), &variances.into()));
                self.filter.update_body_velocity(&velocity, &variances);
            }
            Motion::Pose(mut position, rotation, variances) => {
                // Carried over the time the state has moved on since it was measured.
                position += self.filter.x.velocity * (stamp - measured_at);
                let x = &self.filter.x;
                let turn = super::lio_filter::log(&(x.rotation.transpose() * rotation));
                let offset = position - x.position;
                let residual = [offset[0], offset[1], offset[2], turn[0], turn[1], turn[2]];
                let variances = self.learned(source, &residual, &variances);
                self.filter.update_pose(&position, &rotation, &variances);
            }
            Motion::Imu(index, gyro, accel) => {
                let imu = &mut self.imus[index];
                let (rotation, lever) = imu.mount.expect("pushed once mounted");
                let angular_acceleration = match imu.previous {
                    Some((t0, gyro0)) if measured_at > t0 => rotation * (gyro - gyro0) / (measured_at - t0),
                    _ => Vector3::zeros(),
                };
                imu.previous = Some((measured_at, gyro));
                let mount = ImuMount { rotation, lever, angular_acceleration };
                let (measurement, bias) = (imu.measurement, imu.bias);
                self.filter.update_mounted_imu(&gyro, &accel, &measurement, &mount, bias);
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

    /// Dropped until the lidar's mount is set.
    pub fn push_scan(&mut self, lidar: usize, scan: &LidarScan) {
        if !self.lidar_mounted(lidar) {
            return;
        }
        self.scans_pushed += 1;
        let points = surface_points(scan, self.config.min_range, self.config.keep_every);
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
        let (rotation, _) = self.imus[0].mount.expect("its samples are only taken once mounted");
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
            self.state_at = self.measured[0].stamp;
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
                let (mount_rotation, mount_translation) = self.lidars[lidar].expect("mounted");
                self.map_seed.push(to_world(p, &x.rotation, &x.position, &mount_rotation, &mount_translation));
            }
            if self.map_seed.len() < self.config.map_seed_points {
                return Some(None);
            }
            self.map.insert(&self.map_seed);
            self.map_seed = Vec::new();
            self.map_ready = true;
            self.sync.set_map_ready();
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
        for run in self.runs() {
            let now = self.thinned[start + run - 1].time_ms as f64 / 1000.0 + scan.begin;
            let lidar = self.thinned_lidar[start];
            if !self.first_scan && now < self.state_at {
                // Another lidar's points the filter has already passed.
                self.late_points += run as u64;
                start += run;
                continue;
            }
            self.catch_up_imu(now);
            let (mount_rotation, mount_translation) = self.lidars[lidar].expect("mounted");
            let m = LidarMeasurement { mount_rotation, mount_translation, ..self.config.lidar };
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
        }
        let pose = self.pose(ros_seconds(scan.end));
        if n > 4 {
            self.map.grow(
                &self.world[..n],
                &self.neighbors[..n],
                self.config.map_spacing,
            );
        }
        Some(Some(pose))
    }

    /// Each lidar's points thinned in its own frame, then all of them in time order (a tie
    /// keeps the lidars' order).
    fn thin(&mut self, parts: Vec<(usize, Vec<SurfacePoint>)>) {
        let mut merged: Vec<(SurfacePoint, usize)> = Vec::new();
        for (lidar, points) in parts {
            merged.extend(voxel_centroids(&points, self.config.scan_cube).into_iter().map(|p| (p, lidar)));
        }
        if merged.iter().any(|(_, l)| *l != merged[0].1) {
            merged.sort_by(|a, b| a.0.time_ms.total_cmp(&b.0.time_ms));
        }
        self.thinned = merged.iter().map(|(p, _)| *p).collect();
        self.thinned_lidar = merged.iter().map(|(_, l)| *l).collect();
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
            self.first_scan = false;
            self.covariance_at = now;
            self.state_at = now;
        }
        if !self.sync.imus.is_empty() {
            while now > self.sync.imu_after.stamp {
                let sample = self.sync.imu_after;
                self.take_rates(sample.stamp);
                let previous = (self.state_at, self.rate.0);
                self.rate = (sample.gyro, sample.accel_g);
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
        let imu = &self.imus[0];
        let (rotation, lever) = imu.mount.expect("its samples are only taken once mounted");
        let measurement = imu.measurement;
        let (gyro, accel) = self.rate;
        if rotation == Matrix3::identity() && lever == Vector3::zeros() {
            return self.carrying(|filter| filter.update_imu(&gyro, &accel, &measurement));
        }
        let dt = self.state_at - previous.0;
        let angular_acceleration = if dt > 0.0 {
            rotation * (gyro - previous.1) / dt
        } else {
            Vector3::zeros()
        };
        let mount = ImuMount { rotation, lever, angular_acceleration };
        self.carrying(|filter| filter.update_mounted_imu(&gyro, &accel, &measurement, &mount, BiasAt::Core));
    }
}

/// Inserts `value` into the sorted `values` and returns their median.
fn insert_sorted_median(values: &mut Vec<f64>, value: f64) -> f64 {
    let at = values.partition_point(|r| *r < value);
    values.insert(at, value);
    values[values.len() / 2]
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
