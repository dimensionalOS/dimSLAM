// Copyright 2026 Dimensional Inc.
// SPDX-License-Identifier: Apache-2.0
//
// Lidar-inertial odometry run the way Point-LIO runs it, so the two can be compared frame by
// frame, bit for bit: each scan is thinned and cut into groups of points taken at the same
// instant; before each group the filter predicts up to every IMU sample in between and takes
// that sample as a measurement, then predicts to the group and updates with its points; the
// scan's points then grow the map and the pose at the scan's end is published.

use std::collections::VecDeque;

use nalgebra::{Matrix3, Vector3};

use super::scan_points::{surface_points, voxel_centroids, SurfacePoint};
use super::lio_filter::{
    initial_covariance, process_noise, ImuMeasurement, LidarMeasurement, LioFilter, LioState,
    PointGroup, StateMatrix,
};
use super::plane_fit::time_runs;
use super::point_map::{to_world, PointMap};
use super::scan_sync::{ros_seconds, ScanSync};
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
    pub lidar: LidarMeasurement,
    pub imu: ImuMeasurement,
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
                mount_translation: Vector3::new(-0.011, -0.02329, 0.04412),
            },
            imu: ImuMeasurement {
                gyro_variance: 0.01,
                accel_variance: 0.01,
                clip: Some((35.0, 3.0)),
                accel_scale: 9.81 / 1.0,
            },
            process: [20.0, 1000.0, 500.0, 0.0001, 0.0001],
        }
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

pub struct LioOdometry {
    config: LioConfig,
    pub sync: ScanSync,
    pub map: PointMap,
    pub filter: LioFilter,
    noise: StateMatrix,
    map_seed: Vec<SurfacePoint>,
    map_ready: bool,
    first_scan: bool,
    /// When the state, and the covariance, were last moved forward.
    state_at: f64,
    covariance_at: f64,
    /// The IMU sample the filter is between.
    rate: (Vector3<f64>, Vector3<f64>),
    /// Per point slot, across scans.
    pub world: Vec<SurfacePoint>,
    pub neighbors: Vec<Vec<SurfacePoint>>,
    pub used: Vec<bool>,
    /// The last scan's thinned points, in the IMU frame at their own times.
    pub thinned: Vec<SurfacePoint>,
    scans_pushed: u64,
    /// Motion measured by other sensors, oldest first.
    measured: VecDeque<Measured>,
    /// The state's angular velocity and body-frame velocity at the instants measurements were
    /// last checked for (s, rad/s, m/s), reaching back to the newest one taken: what a late
    /// measurement is measured against.
    motion_history: VecDeque<(f64, Vector3<f64>, Vector3<f64>)>,
    measured_pushed: bool,
    /// Every taken rate's residual against the state, over the spread its given variance
    /// allows, sorted: their median is how much noisier the sensor is than it was said to be.
    rate_residuals: Vec<f64>,
    /// Every taken body velocity's residual per axis (m/s), sorted: their median sets its spread.
    velocity_residuals: Vec<f64>,
}

/// What another sensor measured, body frame.
#[derive(Clone, Copy, Debug)]
enum Motion {
    /// rad/s, with its variance's floor.
    Rate(Vector3<f64>, f64),
    /// m/s.
    Velocity(Vector3<f64>),
}

#[derive(Clone, Copy, Debug)]
struct Measured {
    stamp: f64,
    motion: Motion,
}

impl LioOdometry {
    pub fn new(config: LioConfig) -> Self {
        let [v, w, a, bg, ba] = config.process;
        let map = PointMap::new(config.map_cube, config.map_capacity);
        Self {
            config,
            sync: ScanSync::new(),
            map,
            filter: LioFilter {
                x: LioState::default(),
                p: initial_covariance(),
            },
            noise: process_noise(v, w, a, bg, ba),
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
            scans_pushed: 0,
            measured: VecDeque::new(),
            motion_history: VecDeque::new(),
            measured_pushed: false,
            rate_residuals: Vec::new(),
            velocity_residuals: Vec::new(),
        }
    }

    /// `stamp` in seconds, `gyro` rad/s, `accel` in the accelerometer's unit.
    pub fn push_imu(&mut self, stamp: f64, gyro: Vector3<f64>, accel: Vector3<f64>) {
        self.sync.push_imu(stamp, gyro, accel);
    }

    /// An angular velocity another sensor measured at `stamp` (s), body frame; taken in time
    /// order with the IMU samples.
    pub fn push_rate(&mut self, stamp: f64, rate: Vector3<f64>, variance: f64) {
        self.push_measured(stamp, Motion::Rate(rate, variance));
    }

    /// The body's velocity another sensor measured at `stamp` (s), m/s in the body frame (legged
    /// or wheeled odometry); its spread is learned from how far it sits from the state.
    pub fn push_body_velocity(&mut self, stamp: f64, velocity: Vector3<f64>) {
        self.push_measured(stamp, Motion::Velocity(velocity));
    }

    /// Sensors interleave out of order; each one's own order is kept.
    fn push_measured(&mut self, stamp: f64, motion: Motion) {
        let at = self.measured.partition_point(|m| m.stamp <= stamp);
        self.measured.insert(at, Measured { stamp, motion });
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

    /// Takes the measurements stamped before `until`, each at its own instant. One that arrives
    /// after the state has passed its stamp (a camera's frames trail the IMU) is taken at the
    /// state's instant, carried there by the state's own change since the stamp, so only the
    /// sensor's noise is left between the two.
    fn take_rates(&mut self, until: f64) {
        // Only a LIO given measurements keeps the history; it is trimmed as they are taken.
        if !self.first_scan && self.measured_pushed {
            let body = self.body_velocity();
            self.motion_history
                .push_back((self.state_at, self.filter.x.angular_velocity, body));
        }
        while self.measured.front().is_some_and(|m| m.stamp < until) {
            let Measured { stamp: measured_at, motion } = self.measured.pop_front().expect("checked");
            if self.first_scan {
                continue;
            }
            let late_from = (measured_at < self.state_at)
                .then(|| self.motion_at(measured_at))
                .flatten();
            while self.motion_history.get(1).is_some_and(|(t, _, _)| *t <= measured_at) {
                self.motion_history.pop_front();
            }
            let stamp = measured_at.max(self.state_at);
            self.filter
                .predict(stamp - self.state_at, &self.noise, true, false);
            self.state_at = stamp;
            let since = stamp - self.covariance_at;
            if since > 0.0 {
                self.covariance_at = stamp;
                self.filter.predict(since, &self.noise, false, true);
            }
            match motion {
                Motion::Rate(mut rate, variance) => {
                    if let Some((then, _)) = late_from {
                        rate += self.filter.x.angular_velocity - then;
                    }
                    // The given variance is the sensor's floor; its spread is learned from the
                    // residuals' median absolute value (x1.4826 for a normal's sigma), so
                    // outliers don't set it.
                    let residual =
                        (rate - self.filter.x.angular_velocity).norm() / (3.0 * variance).sqrt();
                    let median = insert_sorted_median(&mut self.rate_residuals, residual);
                    let variance = variance * (1.4826 * median).powi(2).max(1.0);
                    self.filter.update_angular_velocity(&rate, variance);
                }
                Motion::Velocity(mut velocity) => {
                    let body = self.body_velocity();
                    if let Some((_, then)) = late_from {
                        velocity += body - then;
                    }
                    // No floor: the spread is the residuals' alone, per axis.
                    let residual = (velocity - body).norm() / 3f64.sqrt();
                    let median = insert_sorted_median(&mut self.velocity_residuals, residual);
                    let variance = (1.4826 * median).powi(2);
                    if variance > 0.0 {
                        self.filter.update_body_velocity(&velocity, variance);
                    }
                }
            }
        }
    }

    /// Scans handed in so far.
    pub fn scans_pushed(&self) -> u64 {
        self.scans_pushed
    }

    pub fn push_scan(&mut self, scan: &LidarScan) {
        self.scans_pushed += 1;
        let points = surface_points(scan, self.config.min_range, self.config.keep_every);
        self.sync.push_scan(scan.timestamp_ns as u64, points);
    }

    /// Runs the next scan whose IMU samples have all arrived: `None` when there is none, else
    /// whether it published a pose.
    pub fn step(&mut self) -> Option<Option<LioPose>> {
        let scan = self.sync.next_scan()?;
        if let Some(level) = scan.level {
            self.filter.x = LioState {
                rotation: level.rotation,
                gravity: level.gravity,
                specific_force: level.specific_force,
                ..LioState::default()
            };
        }
        if scan.points.is_empty() {
            return Some(None);
        }
        self.thinned = voxel_centroids(&scan.points, self.config.scan_cube);
        let runs = time_runs(&self.thinned);
        let n = self.thinned.len();
        let m = self.config.lidar;
        if !self.map_ready {
            let x = &self.filter.x;
            self.map_seed.extend(self.thinned.iter().map(|p| {
                to_world(
                    p,
                    &x.rotation,
                    &x.position,
                    &m.mount_rotation,
                    &m.mount_translation,
                )
            }));
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
        for &run in &runs {
            let now = self.thinned[start + run - 1].time_ms as f64 / 1000.0 + scan.begin;
            self.catch_up_imu(now);
            let mut group = PointGroup {
                body: &self.thinned[start..start + run],
                world: &mut self.world[start..start + run],
                neighbors: &mut self.neighbors[start..start + run],
                used: &mut self.used[start..start + run],
            };
            if n >= 1 {
                self.filter.update_points(&mut group, &mut self.map, &m);
            }
            start += run;
        }
        let x = &self.filter.x;
        let pose = LioPose {
            stamp: ros_seconds(scan.end),
            position: x.position,
            quaternion: quaternion(&x.rotation),
            velocity: x.velocity,
            angular_velocity: x.angular_velocity,
        };
        if n > 4 {
            self.map.grow(
                &self.world[..n],
                &self.neighbors[..n],
                self.config.map_spacing,
            );
        }
        Some(Some(pose))
    }

    /// Takes every IMU sample before `now` (predicting up to it, then measuring with it) and
    /// predicts the state on to `now`.
    fn catch_up_imu(&mut self, now: f64) {
        let sync = &mut self.sync;
        if self.first_scan {
            while now > sync.imu_after.stamp {
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
                let sync = &mut self.sync;
                self.rate = (sample.gyro, sample.accel_g);
                self.filter
                    .predict(sample.stamp - self.state_at, &self.noise, true, false);
                self.state_at = sample.stamp;
                let since = sample.stamp - self.covariance_at;
                if since > 0.0 {
                    self.covariance_at = sample.stamp;
                    self.filter.predict(since, &self.noise, false, true);
                    self.filter
                        .update_imu(&self.rate.0, &self.rate.1, &self.config.imu);
                }
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
