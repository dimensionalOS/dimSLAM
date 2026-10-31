// Copyright 2026 Dimensional Inc.
// SPDX-License-Identifier: Apache-2.0
//
// Scans and IMU samples grouped the way Point-LIO groups them, and the filter levelled the way
// it levels: so the two can be compared stage by stage, bit for bit. A scan is taken once the
// IMU has reached its last point; until enough samples have been seen, every scan's samples
// join one growing batch whose running mean is the stationary specific force, and the first
// scan past that count turns the body so the mean points against gravity.

use std::collections::VecDeque;

use nalgebra::{Matrix3, Vector3};

use super::scan_points::SurfacePoint;

/// m/s^2 per g, as IMU drivers convert.
pub const STANDARD_GRAVITY: f64 = 9.80665;
/// m/s^2: the gravity the levelled world frame holds.
const WORLD_GRAVITY: f64 = 9.81;
/// The running mean is trusted once it has counted past this many samples.
const LEVEL_SAMPLES: i32 = 100;

/// One IMU sample: seconds as ROS stamps them, rad/s and g.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct StampedImu {
    pub stamp: f64,
    pub gyro: Vector3<f64>,
    pub accel_g: Vector3<f64>,
}

/// What levelling decides: the body's world rotation, the world's gravity and the specific
/// force the body then reads at rest.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Level {
    pub rotation: Matrix3<f64>,
    pub gravity: Vector3<f64>,
    pub specific_force: Vector3<f64>,
}

/// A scan taken with the IMU samples that reach it.
#[derive(Clone, Debug)]
pub struct SyncedScan {
    pub begin: f64,
    pub end: f64,
    pub points: Vec<SurfacePoint>,
    /// Set on the scan that levelled the filter.
    pub level: Option<Level>,
}

/// Seconds rounded to the nanosecond, as a ROS time holds them.
pub fn ros_seconds(t: f64) -> f64 {
    let mut sec = t.floor() as u64;
    let mut nsec = ((t - sec as f64) * 1e9).round() as u64;
    sec += nsec / 1_000_000_000;
    nsec %= 1_000_000_000;
    sec as f64 + 1e-9 * nsec as f64
}

/// A message stamp in ns as seconds, the way a ROS header's sec and nsec add up.
pub fn header_seconds(stamp_ns: i64) -> f64 {
    let sec = stamp_ns.div_euclid(1_000_000_000);
    let nsec = stamp_ns.rem_euclid(1_000_000_000);
    sec as f64 + nsec as f64 * 1e-9
}

#[derive(Debug, Default)]
pub struct ScanSync {
    scans: VecDeque<(f64, Vec<SurfacePoint>)>,
    newest_scan: f64,
    /// Samples not yet taken: the filter's update loop reads them from here once levelled.
    pub imus: VecDeque<StampedImu>,
    newest_imu: f64,
    /// Until the map has its first points, scans still take the samples that reach them.
    map_ready: bool,
    /// The scan at the head of `scans`, with its end, once taken up.
    taken: Option<f64>,
    /// Every sample handed to levelling so far; never emptied, so the mean recounts them.
    batch: Vec<StampedImu>,
    mean_accel: Vector3<f64>,
    mean_gyro: Vector3<f64>,
    counted: i32,
    started: bool,
    levelled: bool,
    /// The samples bracketing the scan start once levelled.
    pub imu_before: StampedImu,
    pub imu_after: StampedImu,
}

impl ScanSync {
    pub fn new() -> Self {
        Self {
            mean_accel: Vector3::new(0.0, 0.0, -1.0),
            counted: 1,
            ..Self::default()
        }
    }

    /// `stamp` in seconds; samples older than the newest are dropped.
    pub fn push_imu(&mut self, stamp: f64, gyro: Vector3<f64>, accel_g: Vector3<f64>) {
        let stamp = ros_seconds(stamp);
        if stamp < self.newest_imu {
            return;
        }
        self.imus.push_back(StampedImu {
            stamp,
            gyro,
            accel_g,
        });
        self.newest_imu = stamp;
    }

    /// `start_ns` the scan's stamp; scans older than the newest are dropped.
    pub fn push_scan(&mut self, start_ns: u64, points: Vec<SurfacePoint>) {
        let stamp = ros_seconds(start_ns as f64 / 1e9);
        if stamp < self.newest_scan {
            return;
        }
        self.newest_scan = stamp;
        self.scans.push_back((stamp, points));
    }

    /// The map has its first points: from here the filter takes the samples itself.
    pub fn set_map_ready(&mut self) {
        self.map_ready = true;
    }

    pub fn pending_scans(&self) -> usize {
        self.scans.len()
    }

    /// The next scan whose samples have all arrived, if any. A scan before levelling comes
    /// back without points: it only fed the mean.
    pub fn next_scan(&mut self) -> Option<SyncedScan> {
        if self.scans.is_empty() || self.imus.is_empty() {
            return None;
        }
        if self.taken.is_none() {
            let (begin, points) = &self.scans[0];
            if points.is_empty() {
                self.scans.pop_front();
                return None;
            }
            // The scan ends at its latest point, the last one counting first.
            let last = points[points.len() - 1].time_ms;
            let latest = points
                .iter()
                .fold(last, |m, p| if p.time_ms > m { p.time_ms } else { m });
            self.taken = Some(begin + latest as f64 / 1000.0);
        }
        let end = self.taken.expect("taken above");
        if self.newest_imu < end {
            return None;
        }
        if !self.levelled || !self.map_ready {
            let mut stamp = self.imus[0].stamp;
            while !self.imus.is_empty() && stamp < end {
                stamp = self.imus[0].stamp;
                if stamp > end {
                    break;
                }
                let sample = self.imus.pop_front().expect("checked non-empty");
                self.batch.push(sample);
                self.imu_before = self.imu_after;
                self.imu_after = sample;
            }
        }
        let (begin, points) = self.scans.pop_front().expect("checked non-empty");
        self.taken = None;
        let level = self.level(begin);
        let points = if self.levelled { points } else { Vec::new() };
        Some(SyncedScan {
            begin,
            end,
            points,
            level,
        })
    }

    fn level(&mut self, begin: f64) -> Option<Level> {
        if self.levelled || self.batch.is_empty() {
            return None;
        }
        if !self.started {
            self.started = true;
            self.counted = 1;
            self.mean_accel = self.batch[0].accel_g;
            self.mean_gyro = self.batch[0].gyro;
        }
        for sample in &self.batch {
            let n = self.counted as f64;
            self.mean_accel += (sample.accel_g - self.mean_accel) / n;
            self.mean_gyro += (sample.gyro - self.mean_gyro) / n;
            self.counted += 1;
        }
        if self.counted <= LEVEL_SAMPLES {
            return None;
        }
        self.levelled = true;
        while begin > self.imu_after.stamp {
            let Some(next) = self.imus.pop_front() else {
                break;
            };
            self.imu_before = self.imu_after;
            self.imu_after = next;
        }
        let measured = (-1.0 * self.mean_accel) * WORLD_GRAVITY / 1.0;
        let gravity = Vector3::new(0.0, 0.0, -WORLD_GRAVITY);
        let rotation = rotation_onto(&measured, &gravity);
        Some(Level {
            rotation,
            gravity,
            specific_force: -rotation.transpose() * gravity,
        })
    }
}

/// The rotation turning `from` onto `onto`'s direction, about their common normal.
fn rotation_onto(from: &Vector3<f64>, onto: &Vector3<f64>) -> Matrix3<f64> {
    let g = onto;
    let cross = Matrix3::new(0.0, g[2], -g[1], -g[2], 0.0, g[0], g[1], -g[0], 0.0);
    let normal = cross * from;
    let sine = normal.norm() / from.norm() / g.norm();
    let cosine = g.dot(from) / g.norm() / from.norm();
    if sine < 1e-6 {
        return if cosine > 1e-6 {
            Matrix3::identity()
        } else {
            -Matrix3::identity()
        };
    }
    let axis_angle = normal / normal.norm() * cosine.acos();
    exp_map(axis_angle[0], axis_angle[1], axis_angle[2])
}

/// Rodrigues' rotation for the axis-angle vector (v1, v2, v3); identity below 1e-5 rad.
fn exp_map(v1: f64, v2: f64, v3: f64) -> Matrix3<f64> {
    let angle = (v1 * v1 + v2 * v2 + v3 * v3).sqrt();
    if angle <= 0.00001 {
        return Matrix3::identity();
    }
    let (x, y, z) = (v1 / angle, v2 / angle, v3 / angle);
    let k = Matrix3::new(0.0, -z, y, z, 0.0, -x, -y, x, 0.0);
    let (sin, cos) = (angle.sin(), angle.cos());
    let scaled = (1.0 - cos) * k;
    let mut rotation = Matrix3::identity();
    for i in 0..3 {
        for j in 0..3 {
            let kk = scaled[(i, 0)] * k[(0, j)]
                + scaled[(i, 1)] * k[(1, j)]
                + scaled[(i, 2)] * k[(2, j)];
            rotation[(i, j)] = rotation[(i, j)] + sin * k[(i, j)] + kk;
        }
    }
    rotation
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn levelling_turns_the_mean_specific_force_onto_up() {
        let tilted = Vector3::new(0.3, -0.5, 0.8).normalize();
        let mut sync = ScanSync::new();
        let point = SurfacePoint {
            position: [1.0, 0.0, 0.0],
            reflectivity: 0.0,
            time_ms: 99.0,
        };
        let mut level = None;
        for scan in 0..30u64 {
            for i in 0..20 {
                let t = scan as f64 * 0.1 + i as f64 * 0.005;
                sync.push_imu(t, Vector3::zeros(), tilted);
            }
            sync.push_scan(scan * 100_000_000, vec![point]);
            while let Some(synced) = sync.next_scan() {
                level = level.or(synced.level);
            }
        }
        let level = level.expect("levelled");
        let up = level.rotation * tilted;
        assert!((up - Vector3::z()).norm() < 1e-12, "{up}");
        assert_eq!(level.gravity, Vector3::new(0.0, 0.0, -WORLD_GRAVITY));
    }

    #[test]
    fn header_and_ros_seconds_round_to_the_nanosecond() {
        assert_eq!(header_seconds(1_500_000_001), 1.0 + 500_000_001.0 * 1e-9);
        assert_eq!(ros_seconds(2.000_000_000_4), 2.0);
    }
}
