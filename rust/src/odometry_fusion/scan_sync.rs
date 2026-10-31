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

/// A scan taken with the IMU samples that reach it, merged with every other lidar's scan that
/// began before it ended: each lidar's points, timed from `begin`.
#[derive(Clone, Debug)]
pub struct SyncedScan {
    pub begin: f64,
    pub end: f64,
    pub parts: Vec<(usize, Vec<SurfacePoint>)>,
    /// Set on the scan that levelled the filter.
    pub level: Option<Level>,
}

/// One lidar's timing, learned from its scans: what tells a lidar that has gone quiet from
/// one whose next scan is merely still in flight.
#[derive(Clone, Copy, Debug, Default)]
struct LidarTiming {
    newest_end: f64,
    scans: u32,
    /// s: mean begin-to-begin interval, and mean time from a scan's end to its arrival, as
    /// the newest IMU stamp measures it.
    mean_period: f64,
    mean_lag: f64,
    newest_begin: f64,
}

impl LidarTiming {
    /// Its next scan is a whole interval later than its usual timing allows.
    fn overdue(&self, now: f64) -> bool {
        self.scans >= 2 && now > self.newest_end + 2.0 * self.mean_period + self.mean_lag
    }
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
    /// Per lidar, its scans' begin and points, oldest first.
    scans: Vec<VecDeque<(f64, Vec<SurfacePoint>)>>,
    timing: Vec<LidarTiming>,
    /// Without an IMU a scan is ready as it arrives and nothing levels.
    pub inertial: bool,
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
    pub fn new(inertial: bool) -> Self {
        Self {
            mean_accel: Vector3::new(0.0, 0.0, -1.0),
            counted: 1,
            inertial,
            levelled: !inertial,
            ..Self::default()
        }
    }

    pub fn newest_imu(&self) -> f64 {
        self.newest_imu
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

    /// `start_ns` the scan's stamp; a lidar's scans older than its newest are dropped.
    pub fn push_scan(&mut self, lidar: usize, start_ns: u64, points: Vec<SurfacePoint>) {
        if self.scans.len() <= lidar {
            self.scans.resize_with(lidar + 1, VecDeque::new);
            self.timing.resize(lidar + 1, LidarTiming::default());
        }
        let stamp = ros_seconds(start_ns as f64 / 1e9);
        let timing = &mut self.timing[lidar];
        if timing.scans > 0 && stamp < timing.newest_begin {
            return;
        }
        let end = stamp + latest_ms(&points) as f64 / 1000.0;
        if timing.scans > 0 {
            let n = timing.scans as f64;
            timing.mean_period += (stamp - timing.newest_begin - timing.mean_period) / n;
            timing.mean_lag += ((self.newest_imu - end).max(0.0) - timing.mean_lag) / n;
        }
        timing.scans += 1;
        timing.newest_begin = stamp;
        timing.newest_end = timing.newest_end.max(end);
        self.scans[lidar].push_back((stamp, points));
    }

    /// The map has its first points: from here the filter takes the samples itself.
    pub fn set_map_ready(&mut self) {
        self.map_ready = true;
    }

    pub fn pending_scans(&self) -> usize {
        self.scans.iter().map(VecDeque::len).sum()
    }

    /// The lidar whose oldest pending scan began first.
    fn head(&self) -> Option<usize> {
        (0..self.scans.len())
            .filter(|&l| !self.scans[l].is_empty())
            .min_by(|&a, &b| self.scans[a][0].0.total_cmp(&self.scans[b][0].0))
    }

    /// The next scan whose samples have all arrived, if any. A scan before levelling comes
    /// back without points: it only fed the mean.
    pub fn next_scan(&mut self) -> Option<SyncedScan> {
        let lead = self.head()?;
        if self.inertial && self.imus.is_empty() {
            return None;
        }
        if self.taken.is_none() {
            let (begin, points) = &self.scans[lead][0];
            if points.is_empty() {
                self.scans[lead].pop_front();
                return None;
            }
            self.taken = Some(begin + latest_ms(points) as f64 / 1000.0);
        }
        let end = self.taken.expect("taken above");
        let now = if self.inertial { self.newest_imu } else { end };
        if self.inertial && self.newest_imu < end {
            return None;
        }
        // Another lidar may still deliver a scan that began before this one ended.
        let waiting = (0..self.scans.len()).any(|l| {
            let timing = &self.timing[l];
            l != lead
                && timing.scans >= 2
                && timing.newest_begin + timing.mean_period < end
                && !timing.overdue(now)
        });
        if waiting && self.inertial {
            return None;
        }
        if self.inertial && (!self.levelled || !self.map_ready) {
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
        let (begin, points) = self.scans[lead].pop_front().expect("checked non-empty");
        self.taken = None;
        let mut parts = vec![(lead, points)];
        for l in 0..self.scans.len() {
            while l != lead && self.scans[l].front().is_some_and(|(b, _)| *b < end) {
                let (other_begin, mut points) = self.scans[l].pop_front().expect("checked");
                let shift = ((other_begin - begin) * 1000.0) as f32;
                for p in &mut points {
                    p.time_ms += shift;
                }
                parts.push((l, points));
            }
        }
        let level = self.level(begin);
        if !self.levelled {
            parts.clear();
        }
        Some(SyncedScan {
            begin,
            end,
            parts,
            level,
        })
    }

    /// Levels from the samples so far without a scan, for a filter with no lidar: true once
    /// levelled.
    pub fn level_without_scans(&mut self) -> Option<Level> {
        if self.levelled || self.imus.is_empty() {
            return None;
        }
        let newest = self.newest_imu;
        while let Some(sample) = self.imus.pop_front() {
            self.batch.push(sample);
            self.imu_before = self.imu_after;
            self.imu_after = sample;
        }
        self.level(newest)
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

/// ms: a scan's latest point, the last one counting first.
fn latest_ms(points: &[SurfacePoint]) -> f32 {
    let Some(last) = points.last() else { return 0.0 };
    points.iter().fold(last.time_ms, |m, p| if p.time_ms > m { p.time_ms } else { m })
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
        let mut sync = ScanSync::new(true);
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
            sync.push_scan(0, scan * 100_000_000, vec![point]);
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
