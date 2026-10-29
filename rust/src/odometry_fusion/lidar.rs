// Copyright 2026 Dimensional Inc.
// SPDX-License-Identifier: Apache-2.0
//
// Lidar as a direct measurement of the fused state. A scan is range-gated, thinned to one point
// per voxel and cut into short time slices; each slice is an event of its own, so the filter
// steps to it like any other measurement and the IMU carries the pose between slices. A
// slice's points are matched against a map of planes, one per voxel, each kept as the running
// mean and scatter of the points that fell in it: a lookup is a hash and a 3x3 eigensolve, with
// no neighbor search. Points enter the map only once their slice has left the replay buffer,
// at the pose the filter settled on, since a map cannot be un-inserted when a late message
// rewrites history.

use std::collections::{HashMap, HashSet};

use nalgebra::{Isometry3, Matrix3, Matrix6, SymmetricEigen, UnitQuaternion, Vector3, Vector6};
use serde::{Deserialize, Serialize};

use crate::types::LidarScan;

/// One physical lidar.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct LidarConfig {
    /// The frame_id its scans carry; scans from any other frame are dropped.
    pub frame_id: String,
    /// m, from the sensor. Points nearer are the robot itself or too noisy to use.
    pub min_range: f64,
    /// m. Points farther are too sparse to make planes of and lever every angular error.
    pub max_range: f64,
    /// m. A scan keeps one point per voxel of this size, its earliest.
    pub scan_voxel_size: f64,
    /// s. A scan becomes one measurement per slice of this length: shorter tracks faster motion
    /// between slices, longer gives each update more points.
    pub slice_seconds: f64,
    /// m. Range noise of one point.
    pub point_std: f64,
}

impl Default for LidarConfig {
    fn default() -> Self {
        Self {
            frame_id: String::new(),
            min_range: 0.5,
            max_range: 60.0,
            scan_voxel_size: 0.3,
            slice_seconds: 0.01,
            point_std: 0.02,
        }
    }
}

/// The plane map every lidar matches against and builds.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct LidarMapConfig {
    /// m. Each voxel holds one plane fit, so this is about the size of the smallest flat patch
    /// the map can use.
    pub voxel_size: f64,
    /// A voxel stops taking points once it holds this many: the plane is settled by then, and
    /// points from a pose that later drifted would only thicken it.
    pub max_points_per_voxel: u32,
    /// Points a voxel needs before its plane is used.
    pub min_points_per_plane: u32,
    /// m. A voxel whose points are thicker than this (std across the plane) is not a plane.
    pub max_plane_std: f64,
    /// A voxel whose points are not this many times wider within the plane than across it
    /// (both as std) is a line or a blob, with no normal to trust.
    pub min_plane_aspect: f64,
    /// m. A voxel whose points spread less than this (std) along the plane's narrower axis
    /// is a thin strip: noise-free or not, its normal could point anywhere around the strip.
    pub min_plane_spread: f64,
    /// A point whose squared distance from its plane, over that distance's predicted variance
    /// (point noise, plane thickness and the pose's own uncertainty), is past this is a
    /// mismatch and is left out.
    pub point_gate: f64,
    /// m. Voxels farther than this from the body are dropped.
    pub radius: f64,
}

impl Default for LidarMapConfig {
    fn default() -> Self {
        Self {
            voxel_size: 0.5,
            max_points_per_voxel: 50,
            min_points_per_plane: 5,
            max_plane_std: 0.05,
            min_plane_aspect: 3.0,
            min_plane_spread: 0.05,
            point_gate: 9.0,
            radius: 100.0,
        }
    }
}

/// A slice point in the body (output) frame, `dt` seconds after the slice's stamp (<= 0).
#[derive(Clone, Copy, Debug)]
pub struct SlicePoint {
    pub body: Vector3<f32>,
    pub dt: f32,
}

#[derive(Clone, Debug)]
pub struct CloudSlice {
    pub lidar: usize,
    pub points: Vec<SlicePoint>,
}

/// Range-gates, thins and slices one scan; each slice is stamped at its last point.
pub fn slice_scan(
    scan: &LidarScan,
    lidar: usize,
    config: &LidarConfig,
    base_from_lidar: &Isometry3<f64>,
) -> Vec<(i64, CloudSlice)> {
    let timed = scan.offsets_ns.len() == scan.points.len();
    let inverse_voxel = 1.0 / config.scan_voxel_size;
    let mut occupied = HashSet::new();
    let mut kept: Vec<(i64, Vector3<f64>)> = Vec::new();
    let mut order: Vec<usize> = (0..scan.points.len()).collect();
    if timed {
        order.sort_by_key(|&i| scan.offsets_ns[i]);
    }
    for i in order {
        let [x, y, z] = scan.points[i];
        let point = Vector3::new(x as f64, y as f64, z as f64);
        let range = point.norm();
        if !range.is_finite() || range < config.min_range || range > config.max_range {
            continue;
        }
        let body = base_from_lidar * nalgebra::Point3::from(point);
        let key = voxel_key(&body.coords, inverse_voxel);
        if !occupied.insert(key) {
            continue;
        }
        let offset = if timed { scan.offsets_ns[i] as i64 } else { 0 };
        kept.push((scan.timestamp_ns + offset, body.coords));
    }
    let slice_ns = (config.slice_seconds * 1.0e9).max(1.0) as i64;
    let mut slices = Vec::new();
    let mut start = 0;
    while start < kept.len() {
        let limit = kept[start].0 + slice_ns;
        let end = start + kept[start..].partition_point(|(ts, _)| *ts < limit);
        let stamp = kept[end - 1].0;
        let points = kept[start..end]
            .iter()
            .map(|(ts, body)| SlicePoint {
                body: body.cast(),
                dt: ((ts - stamp) as f64 / 1.0e9) as f32,
            })
            .collect();
        slices.push((stamp, CloudSlice { lidar, points }));
        start = end;
    }
    slices
}

fn voxel_key(point: &Vector3<f64>, inverse_size: f64) -> [i32; 3] {
    [
        (point.x * inverse_size).floor() as i32,
        (point.y * inverse_size).floor() as i32,
        (point.z * inverse_size).floor() as i32,
    ]
}

/// A point's world position at `dt` from the state's stamp, moved back along the state's own
/// velocity and rate: the slice's points are spread over its length, the state is at its end.
pub fn world_point(
    point: &SlicePoint,
    position: &Vector3<f64>,
    q: &UnitQuaternion<f64>,
    velocity: &Vector3<f64>,
    angular_velocity: &Vector3<f64>,
) -> (Vector3<f64>, Vector3<f64>) {
    let dt = point.dt as f64;
    let body = UnitQuaternion::from_scaled_axis(angular_velocity * dt) * point.body.cast::<f64>();
    (q * body + position + velocity * dt, body)
}

#[derive(Clone, Copy, Debug)]
pub struct Plane {
    /// Unit normal and offset: a point x is on the plane when normal . x + offset = 0.
    pub normal: Vector3<f64>,
    pub offset: f64,
    /// The points' variance across the plane.
    pub variance: f64,
}

#[derive(Clone, Debug, Default)]
struct Voxel {
    count: u32,
    mean: Vector3<f64>,
    /// Sum of outer products of deviations from the mean (Welford).
    scatter: Matrix3<f64>,
    /// The fit for the current `count`; None until fitted, or when not a plane.
    plane: Option<Plane>,
    fitted_count: u32,
}

pub struct PlaneMap {
    config: LidarMapConfig,
    inverse_size: f64,
    voxels: HashMap<[i32; 3], Voxel>,
    /// Where the body was at the last prune.
    pruned_at: Option<Vector3<f64>>,
}

impl PlaneMap {
    pub fn new(config: LidarMapConfig) -> Self {
        let inverse_size = 1.0 / config.voxel_size;
        Self {
            config,
            inverse_size,
            voxels: HashMap::new(),
            pruned_at: None,
        }
    }

    pub fn config(&self) -> &LidarMapConfig {
        &self.config
    }

    pub fn voxel_count(&self) -> usize {
        self.voxels.len()
    }

    pub fn insert(&mut self, point: &Vector3<f64>) {
        let key = voxel_key(point, self.inverse_size);
        let voxel = self.voxels.entry(key).or_default();
        if voxel.count >= self.config.max_points_per_voxel {
            return;
        }
        voxel.count += 1;
        let delta = point - voxel.mean;
        voxel.mean += delta / voxel.count as f64;
        voxel.scatter += delta * (point - voxel.mean).transpose();
    }

    /// The plane through the voxel `point` falls in, when that voxel holds one.
    pub fn plane_at(&mut self, point: &Vector3<f64>) -> Option<Plane> {
        let key = voxel_key(point, self.inverse_size);
        let config = &self.config;
        let voxel = self.voxels.get_mut(&key)?;
        if voxel.count < config.min_points_per_plane {
            return None;
        }
        if voxel.fitted_count != voxel.count {
            voxel.fitted_count = voxel.count;
            voxel.plane = fit_plane(voxel, config);
        }
        voxel.plane
    }

    /// Drops the voxels past `radius`, once the body has moved a voxel since the last prune.
    pub fn prune(&mut self, body: &Vector3<f64>) {
        if let Some(at) = self.pruned_at {
            if (body - at).norm() < self.config.voxel_size {
                return;
            }
        }
        self.pruned_at = Some(*body);
        let radius_sq = self.config.radius * self.config.radius;
        self.voxels
            .retain(|_, voxel| (voxel.mean - body).norm_squared() <= radius_sq);
    }
}

fn fit_plane(voxel: &Voxel, config: &LidarMapConfig) -> Option<Plane> {
    let covariance = voxel.scatter / voxel.count as f64;
    let eigen = SymmetricEigen::new(covariance);
    let mut order = [0, 1, 2];
    order.sort_by(|a, b| eigen.eigenvalues[*a].total_cmp(&eigen.eigenvalues[*b]));
    let thickness = eigen.eigenvalues[order[0]].max(0.0);
    let spread = eigen.eigenvalues[order[1]];
    let aspect_sq = config.min_plane_aspect * config.min_plane_aspect;
    let too_thick = thickness > config.max_plane_std * config.max_plane_std;
    let too_narrow = spread <= aspect_sq * thickness
        || spread < config.min_plane_spread * config.min_plane_spread;
    if too_thick || too_narrow {
        return None;
    }
    let normal: Vector3<f64> = eigen.eigenvectors.column(order[0]).into_owned();
    Some(Plane {
        normal,
        offset: -normal.dot(&voxel.mean),
        variance: thickness,
    })
}

/// A slice's matched points summarized for one update of the pose columns: the information
/// sum(h h^T / s^2), the information-weighted residual sum(h r / s^2), sum(r^2 / s^2), and
/// sum(ln s^2), over the points that passed the gate. h is a point's row over
/// [position, rotation], r its distance to its plane, s^2 its noise.
#[derive(Clone, Debug, Default)]
pub struct PlaneEvidence {
    pub information: Matrix6<f64>,
    pub weighted_residual: Vector6<f64>,
    pub weighted_residual_sq: f64,
    pub log_det_noise: f64,
    pub matched: usize,
    pub gated: usize,
}

impl PlaneEvidence {
    /// Adds one point-to-plane row: `row` over [position, rotation], `residual` the plane
    /// distance to remove, `noise` its variance, `pose_covariance` the pose block of P.
    pub fn add(
        &mut self,
        row: &Vector6<f64>,
        residual: f64,
        noise: f64,
        pose_covariance: &Matrix6<f64>,
        gate: f64,
    ) {
        let predicted = (row.transpose() * pose_covariance * row)[(0, 0)] + noise;
        if residual * residual > gate * predicted {
            self.gated += 1;
            return;
        }
        self.information += row * row.transpose() / noise;
        self.weighted_residual += row * (residual / noise);
        self.weighted_residual_sq += residual * residual / noise;
        self.log_det_noise += noise.ln();
        self.matched += 1;
    }
}

/// The point-to-plane row of `world` (seen at body-frame `body`) under the right-perturbed
/// rotation `q`: d(n . x)/d[position, theta] = [n, -(R [body]x)^T n].
pub fn plane_row(plane: &Plane, q: &UnitQuaternion<f64>, body: &Vector3<f64>) -> Vector6<f64> {
    let rotation_row = -(q.to_rotation_matrix().matrix() * body.cross_matrix()).transpose()
        * plane.normal;
    let mut row = Vector6::zeros();
    row.fixed_rows_mut::<3>(0).copy_from(&plane.normal);
    row.fixed_rows_mut::<3>(3).copy_from(&rotation_row);
    row
}

#[cfg(test)]
mod tests {
    use super::*;

    fn scan(points: Vec<[f32; 3]>, offsets_ns: Vec<u32>) -> LidarScan {
        LidarScan {
            timestamp_ns: 1_000_000_000,
            frame_id: "lidar".into(),
            points,
            offsets_ns,
        }
    }

    #[test]
    fn a_scan_is_range_gated_thinned_and_cut_into_slices_stamped_at_their_last_point() {
        let config = LidarConfig {
            min_range: 1.0,
            max_range: 10.0,
            scan_voxel_size: 0.5,
            slice_seconds: 0.01,
            ..LidarConfig::default()
        };
        let points = vec![
            [0.5, 0.0, 0.0],  // too near
            [2.0, 0.0, 0.0],  // slice 0
            [2.1, 0.0, 0.0],  // same voxel as the one before
            [3.0, 0.0, 0.0],  // slice 0
            [20.0, 0.0, 0.0], // too far
            [4.0, 0.0, 0.0],  // slice 1
        ];
        let offsets = vec![0, 1_000_000, 2_000_000, 5_000_000, 6_000_000, 15_000_000];
        let slices = slice_scan(&scan(points, offsets), 0, &config, &Isometry3::identity());
        assert_eq!(slices.len(), 2);
        assert_eq!(slices[0].0, 1_005_000_000);
        assert_eq!(slices[0].1.points.len(), 2);
        assert!((slices[0].1.points[0].dt + 0.004).abs() < 1e-6);
        assert_eq!(slices[0].1.points[1].dt, 0.0);
        assert_eq!(slices[1].0, 1_015_000_000);
        assert_eq!(slices[1].1.points.len(), 1);
    }

    #[test]
    fn scan_points_are_carried_into_the_body_frame() {
        let mount = Isometry3::translation(0.0, 0.0, 1.0);
        let slices = slice_scan(
            &scan(vec![[2.0, 0.0, 0.0]], vec![0]),
            0,
            &LidarConfig::default(),
            &mount,
        );
        let point = slices[0].1.points[0].body;
        assert!((point - Vector3::new(2.0, 0.0, 1.0)).norm() < 1e-6);
    }

    /// A 10x10 grid at 5 cm on z, in a scattered order so any prefix spans the square.
    fn floor(map: &mut PlaneMap, z: f64) {
        for k in 0..100 {
            let cell = (k * 37) % 100;
            map.insert(&Vector3::new(0.05 * (cell / 10) as f64, 0.05 * (cell % 10) as f64, z));
        }
    }

    #[test]
    fn a_flat_voxel_fits_its_plane_and_a_line_does_not() {
        let mut map = PlaneMap::new(LidarMapConfig {
            max_points_per_voxel: 1000,
            ..LidarMapConfig::default()
        });
        floor(&mut map, 0.2);
        let plane = map.plane_at(&Vector3::new(0.1, 0.1, 0.1)).expect("a plane");
        assert!(plane.normal.z.abs() > 0.999);
        assert!((plane.normal.dot(&Vector3::new(0.3, 0.3, 0.2)) + plane.offset).abs() < 1e-9);

        for i in 0..20 {
            map.insert(&Vector3::new(1.0 + 0.02 * i as f64, 1.2, 1.2));
        }
        assert!(map.plane_at(&Vector3::new(1.1, 1.2, 1.2)).is_none());
    }

    #[test]
    fn a_full_voxel_stops_taking_points() {
        let mut map = PlaneMap::new(LidarMapConfig {
            max_points_per_voxel: 20,
            ..LidarMapConfig::default()
        });
        floor(&mut map, 0.2);
        let before = map.plane_at(&Vector3::new(0.1, 0.1, 0.2)).unwrap();
        for _ in 0..100 {
            map.insert(&Vector3::new(0.1, 0.1, 0.45));
        }
        let after = map.plane_at(&Vector3::new(0.1, 0.1, 0.2)).unwrap();
        assert_eq!(before.offset, after.offset);
    }

    #[test]
    fn prune_drops_voxels_past_the_radius() {
        let mut map = PlaneMap::new(LidarMapConfig {
            radius: 5.0,
            ..LidarMapConfig::default()
        });
        map.insert(&Vector3::new(1.0, 0.0, 0.0));
        map.insert(&Vector3::new(9.0, 0.0, 0.0));
        map.prune(&Vector3::zeros());
        assert_eq!(map.voxel_count(), 1);
    }

    #[test]
    fn the_plane_row_matches_a_finite_difference() {
        let plane = Plane {
            normal: Vector3::new(0.3, -0.4, 0.866).normalize(),
            offset: 0.7,
            variance: 0.0,
        };
        let q = UnitQuaternion::from_euler_angles(0.2, -0.1, 0.7);
        let position = Vector3::new(1.0, 2.0, 0.5);
        let body = Vector3::new(3.0, -1.0, 0.4);
        let distance = |position: &Vector3<f64>, q: &UnitQuaternion<f64>| {
            plane.normal.dot(&(q * body + position)) + plane.offset
        };
        let row = plane_row(&plane, &q, &body);
        let eps = 1e-6;
        for axis in 0..3 {
            let mut step = Vector3::zeros();
            step[axis] = eps;
            let dp = (distance(&(position + step), &q) - distance(&position, &q)) / eps;
            assert!((dp - row[axis]).abs() < 1e-5);
            let dq = (distance(&position, &(q * UnitQuaternion::from_scaled_axis(step)))
                - distance(&position, &q))
                / eps;
            assert!((dq - row[3 + axis]).abs() < 1e-5, "theta {axis}: {dq} vs {}", row[3 + axis]);
        }
    }

    #[test]
    fn the_gate_counts_rather_than_fuses_a_far_point() {
        let mut evidence = PlaneEvidence::default();
        let row = Vector6::new(0.0, 0.0, 1.0, 0.0, 0.0, 0.0);
        let p = Matrix6::identity() * 1e-4;
        evidence.add(&row, 0.01, 1e-4, &p, 9.0);
        evidence.add(&row, 1.0, 1e-4, &p, 9.0);
        assert_eq!((evidence.matched, evidence.gated), (1, 1));
    }
}
