// Copyright 2026 Dimensional Inc.
// SPDX-License-Identifier: Apache-2.0
//
// Lidar as a direct measurement of the fused state. A scan is range-gated, thinned to one point
// per voxel and cut into short time slices; each slice is an event of its own, so the filter
// steps to it like any other measurement and the IMU carries the pose between slices. A
// slice's points are matched against a map of planes, one per voxel, each kept as the running
// mean and scatter of the points that fell in it: a lookup is a hash and a 3x3 eigensolve, with
// no neighbor search. A slice's points enter the map right after its own update; a late
// message that replays it later cannot move them, since a map cannot be un-inserted.

use std::collections::{HashMap, HashSet};

use nalgebra::{DMatrix, DVector, Isometry3, Matrix3, SymmetricEigen, UnitQuaternion, Vector3};
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
    /// m. Each slice keeps one point per voxel of this size, its earliest.
    pub scan_voxel_size: f64,
    /// s. A scan becomes one measurement per slice of this length: shorter tracks faster motion
    /// between slices, longer gives each update more points.
    pub slice_seconds: f64,
    /// s, added to every scan's stamp: the lidar clock's lag behind the IMU's, or a driver that
    /// stamps a sweep at its end rather than its start (then minus the sweep's length).
    pub time_offset_seconds: f64,
    /// rad. How well `base_from_lidar`'s rotation is known: above zero the filter estimates a
    /// correction to it from the scans, starting this uncertain. Off by default: on the
    /// benchmarks the estimate settled on a correction that hurt rather than helped. A mount off by a degree
    /// lifts or sinks the body with every meter walked, since the map is built on the scans.
    pub mount_rotation_std: f64,
    /// m. Noise of one point-to-plane distance. Not the sensor's range noise alone: plane
    /// fits and the map's own pose errors are shared by every point of a slice, so this is set
    /// well above it, or a slice's hundred points make the filter overconfident and its gate
    /// then refuses the very points that would correct it.
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
            time_offset_seconds: 0.0,
            mount_rotation_std: 0.0,
            point_std: 0.05,
        }
    }
}

/// The plane map every lidar matches against and builds.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct LidarMapConfig {
    /// How a point finds the surface it hit: a plane per voxel, or a plane through its
    /// nearest map points.
    pub matching: Matching,
    /// `nearest_points`: how many map points a point's plane is fitted through.
    pub neighbors: u32,
    /// m. `nearest_points`: a point joins the map only where no map point is this close.
    pub point_spacing: f64,
    /// m. Each voxel holds one plane fit (`voxel_planes`), or is one cell of the point map
    /// searched with its neighbors, which bounds how far a neighbor can be (`nearest_points`). A point is matched to the largest voxel around it
    /// that is flat, so walls and floors get big, steady planes.
    pub voxel_size: f64,
    /// Voxel sizes, each half the one before, starting at `voxel_size`: where a voxel is not
    /// flat (stairs, furniture, corners) its point falls through to a smaller one that is.
    pub levels: u32,
    /// A voxel (or `nearest_points` cell) stops taking points once it holds this many: the
    /// surface is settled by then, and points from a pose that later drifted would only smear it.
    pub max_points_per_voxel: u32,
    /// Points a voxel needs before its plane is used.
    pub min_points_per_plane: u32,
    /// m. A voxel whose points are thicker than this (std across the plane) is not a plane.
    pub max_plane_std: f64,
    /// A voxel whose points are not this many times wider within the plane than across it
    /// (both as std) is a line or a blob, with no normal to trust.
    pub min_plane_aspect: f64,
    /// A voxel whose points spread less than this fraction of its size (std) along the
    /// plane's narrower axis is a thin strip: noise-free or not, its normal could point
    /// anywhere around the strip.
    pub min_plane_spread: f64,
    /// A point whose squared distance from its plane, over that distance's predicted variance
    /// (point noise, plane thickness and the pose's own uncertainty), is past this is a
    /// mismatch and is left out.
    pub point_gate: f64,
    /// Updates per slice, each re-matching the points from the pose the last one left: a
    /// slice that moves the pose far enough to change which plane a point falls in gets
    /// matched again from there.
    pub iterations: u32,
    /// m. Voxels farther than this from the body are dropped.
    pub radius: f64,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Matching {
    VoxelPlanes,
    #[default]
    NearestPoints,
}

impl Default for LidarMapConfig {
    fn default() -> Self {
        Self {
            matching: Matching::NearestPoints,
            neighbors: 5,
            point_spacing: 0.05,
            voxel_size: 0.5,
            levels: 3,
            max_points_per_voxel: 50,
            min_points_per_plane: 5,
            max_plane_std: 0.05,
            min_plane_aspect: 3.0,
            min_plane_spread: 0.1,
            point_gate: 9.0,
            iterations: 3,
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
    /// The lidar's origin in the body frame: a mount correction turns the points about it.
    pub origin: Vector3<f64>,
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
    let mut gated: Vec<(i64, Vector3<f64>)> = Vec::new();
    for (i, [x, y, z]) in scan.points.iter().enumerate() {
        let point = Vector3::new(*x as f64, *y as f64, *z as f64);
        let range = point.norm();
        if !range.is_finite() || range < config.min_range || range > config.max_range {
            continue;
        }
        let offset = if timed { scan.offsets_ns[i] as i64 } else { 0 };
        let body = base_from_lidar * nalgebra::Point3::from(point);
        let shift = (config.time_offset_seconds * 1.0e9) as i64;
        gated.push((scan.timestamp_ns + shift + offset, body.coords));
    }
    gated.sort_by_key(|(ts, _)| *ts);
    // Thinned per slice, not per scan: per scan, the earliest point in each voxel wins and the
    // scan's late slices are left with almost nothing.
    let inverse_voxel = 1.0 / config.scan_voxel_size;
    let slice_ns = (config.slice_seconds * 1.0e9).max(1.0) as i64;
    let mut slices = Vec::new();
    let mut start = 0;
    while start < gated.len() {
        let limit = gated[start].0 + slice_ns;
        let end = start + gated[start..].partition_point(|(ts, _)| *ts < limit);
        let stamp = gated[end - 1].0;
        let mut occupied = HashSet::new();
        let points = gated[start..end]
            .iter()
            .filter(|(_, body)| occupied.insert(voxel_key(body, inverse_voxel)))
            .map(|(ts, body)| SlicePoint {
                body: body.cast(),
                dt: ((ts - stamp) as f64 / 1.0e9) as f32,
            })
            .collect();
        slices.push((
            stamp,
            CloudSlice {
                lidar,
                origin: base_from_lidar.translation.vector,
                points,
            },
        ));
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

/// Where a slice point lands: its body position with the mount correction `mount` turned in
/// about the lidar's `origin`, moved back along the state's own velocity and rate to the
/// point's time (the slice's points spread over its length, the state is at its end).
pub struct Landed {
    pub world: Vector3<f64>,
    /// The body position at the point's time, deskewed.
    pub body: Vector3<f64>,
    /// world_from_body rotation at the point's time, times the deskew: what turns a body
    /// perturbation into a world one.
    pub rotation: Matrix3<f64>,
    /// The point relative to the lidar origin after the mount correction, before deskew.
    pub from_origin: Vector3<f64>,
}

pub fn land(
    point: &SlicePoint,
    origin: &Vector3<f64>,
    mount: &UnitQuaternion<f64>,
    position: &Vector3<f64>,
    q: &UnitQuaternion<f64>,
    velocity: &Vector3<f64>,
    angular_velocity: &Vector3<f64>,
) -> Landed {
    let dt = point.dt as f64;
    let from_origin = mount * (point.body.cast::<f64>() - origin);
    let deskew = UnitQuaternion::from_scaled_axis(angular_velocity * dt);
    let body = deskew * (from_origin + origin);
    Landed {
        world: q * body + position + velocity * dt,
        body,
        rotation: (q * deskew).to_rotation_matrix().into_inner(),
        from_origin,
    }
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
    /// `nearest_points`: the map's points, by cell of `voxel_size`.
    cells: HashMap<[i32; 3], Vec<Vector3<f64>>>,
    /// Per level, coarsest first: 1 / voxel size and the voxels.
    levels: Vec<(f64, HashMap<[i32; 3], Voxel>)>,
    /// Where the body was at the last prune.
    pruned_at: Option<Vector3<f64>>,
}

impl PlaneMap {
    pub fn new(config: LidarMapConfig) -> Self {
        let levels = (0..config.levels.max(1))
            .map(|level| ((1u32 << level) as f64 / config.voxel_size, HashMap::new()))
            .collect();
        Self {
            config,
            cells: HashMap::new(),
            levels,
            pruned_at: None,
        }
    }

    pub fn config(&self) -> &LidarMapConfig {
        &self.config
    }

    /// Voxels at the coarsest level.
    pub fn voxel_count(&self) -> usize {
        self.levels[0].1.len() + self.cells.len()
    }

    pub fn insert(&mut self, point: &Vector3<f64>) {
        if self.config.matching == Matching::NearestPoints {
            // Capped like a plane voxel: the map keeps the points it was first built from,
            // rather than smearing in later ones from a pose that has since drifted.
            let spacing_sq = self.config.point_spacing * self.config.point_spacing;
            let cap = self.config.max_points_per_voxel as usize;
            let cell = self.cells.entry(voxel_key(point, self.levels[0].0)).or_default();
            if cell.len() < cap && cell.iter().all(|p| (p - point).norm_squared() > spacing_sq) {
                cell.push(*point);
            }
            return;
        }
        let cap = self.config.max_points_per_voxel;
        for (inverse_size, voxels) in &mut self.levels {
            let voxel = voxels.entry(voxel_key(point, *inverse_size)).or_default();
            if voxel.count >= cap {
                continue;
            }
            voxel.count += 1;
            let delta = point - voxel.mean;
            voxel.mean += delta / voxel.count as f64;
            voxel.scatter += delta * (point - voxel.mean).transpose();
        }
    }

    /// The plane of the largest flat voxel `point` falls in, or through its nearest points.
    pub fn plane_at(&mut self, point: &Vector3<f64>) -> Option<Plane> {
        if self.config.matching == Matching::NearestPoints {
            return self.nearest_plane(point);
        }
        let config = &self.config;
        for (inverse_size, voxels) in &mut self.levels {
            let Some(voxel) = voxels.get_mut(&voxel_key(point, *inverse_size)) else {
                // An empty voxel's sub-voxels are empty too.
                return None;
            };
            if voxel.count < config.min_points_per_plane {
                return None;
            }
            if voxel.fitted_count != voxel.count {
                voxel.fitted_count = voxel.count;
                voxel.plane = fit_plane(voxel, config, 1.0 / *inverse_size);
            }
            if voxel.plane.is_some() {
                return voxel.plane;
            }
        }
        None
    }

    /// The plane through the `neighbors` map points nearest `point` within a cell of it, when
    /// they are flat: none farther from their fit than twice `max_plane_std`.
    fn nearest_plane(&self, point: &Vector3<f64>) -> Option<Plane> {
        let config = &self.config;
        let wanted = config.neighbors.max(3) as usize;
        let size = config.voxel_size;
        let inverse_size = self.levels[0].0;
        let key = voxel_key(point, inverse_size);
        // Within its cell, the point's distance to each face: a neighbor cell can hold nothing
        // nearer than the gap to it, so cells are visited nearest first and the search stops
        // at the first one farther than the worst of the best found so far.
        let low = Vector3::new(key[0] as f64, key[1] as f64, key[2] as f64) * size;
        let below = point - low;
        let above = Vector3::from_element(size) - below;
        let mut cells: [(f64, [i32; 3]); 27] = [(0.0, [0; 3]); 27];
        let mut n = 0;
        for dx in -1..=1i32 {
            for dy in -1..=1i32 {
                for dz in -1..=1i32 {
                    let gap = |offset: i32, axis: usize| match offset {
                        -1 => below[axis],
                        1 => above[axis],
                        _ => 0.0,
                    };
                    let gap_sq = gap(dx, 0).powi(2) + gap(dy, 1).powi(2) + gap(dz, 2).powi(2);
                    cells[n] = (gap_sq, [key[0] + dx, key[1] + dy, key[2] + dz]);
                    n += 1;
                }
            }
        }
        cells.sort_unstable_by(|a, b| a.0.total_cmp(&b.0));
        let reach_sq = size * size;
        // The best `wanted` so far, nearest first.
        let mut nearest: Vec<(f64, Vector3<f64>)> = Vec::with_capacity(wanted + 1);
        for (gap_sq, cell_key) in cells {
            let worst = if nearest.len() == wanted {
                nearest[wanted - 1].0
            } else {
                reach_sq
            };
            if gap_sq > worst {
                break;
            }
            let Some(cell) = self.cells.get(&cell_key) else {
                continue;
            };
            for p in cell {
                let d = (p - point).norm_squared();
                let worst = if nearest.len() == wanted {
                    nearest[wanted - 1].0
                } else {
                    reach_sq
                };
                if d > worst {
                    continue;
                }
                let at = nearest.partition_point(|(e, _)| *e <= d);
                nearest.insert(at, (d, *p));
                nearest.truncate(wanted);
            }
        }
        if nearest.len() < wanted {
            return None;
        }
        let mean: Vector3<f64> =
            nearest.iter().map(|(_, p)| p).sum::<Vector3<f64>>() / wanted as f64;
        let mut covariance = Matrix3::zeros();
        for (_, p) in &nearest {
            let d = p - mean;
            covariance += d * d.transpose();
        }
        covariance /= wanted as f64;
        let eigen = SymmetricEigen::new(covariance);
        let mut order = [0, 1, 2];
        order.sort_by(|a, b| eigen.eigenvalues[*a].total_cmp(&eigen.eigenvalues[*b]));
        // Collinear neighbors have no normal.
        if eigen.eigenvalues[order[1]] <= config.min_plane_aspect.powi(2) * eigen.eigenvalues[order[0]].max(0.0)
        {
            return None;
        }
        let normal: Vector3<f64> = eigen.eigenvectors.column(order[0]).into_owned();
        let offset = -normal.dot(&mean);
        let limit = 2.0 * config.max_plane_std;
        if nearest
            .iter()
            .any(|(_, p)| (normal.dot(p) + offset).abs() > limit)
        {
            return None;
        }
        Some(Plane {
            normal,
            offset,
            variance: eigen.eigenvalues[order[0]].max(0.0),
        })
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
        for (_, voxels) in &mut self.levels {
            voxels.retain(|_, voxel| (voxel.mean - body).norm_squared() <= radius_sq);
        }
        self.cells
            .retain(|_, cell| cell.first().is_none_or(|p| (p - body).norm_squared() <= radius_sq));
    }
}

fn fit_plane(voxel: &Voxel, config: &LidarMapConfig, size: f64) -> Option<Plane> {
    let covariance = voxel.scatter / voxel.count as f64;
    let eigen = SymmetricEigen::new(covariance);
    let mut order = [0, 1, 2];
    order.sort_by(|a, b| eigen.eigenvalues[*a].total_cmp(&eigen.eigenvalues[*b]));
    let thickness = eigen.eigenvalues[order[0]].max(0.0);
    let spread = eigen.eigenvalues[order[1]];
    let aspect_sq = config.min_plane_aspect * config.min_plane_aspect;
    let too_thick = thickness > config.max_plane_std * config.max_plane_std;
    let too_narrow = spread <= aspect_sq * thickness
        || spread < (config.min_plane_spread * size).powi(2);
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

/// A slice's matched points summarized for one update of the pose (and mount) columns: the
/// information sum(h h^T / s^2), the information-weighted residual sum(h r / s^2),
/// sum(r^2 / s^2), and sum(ln s^2), over the points that passed the gate. h is a point's row,
/// r its distance to its plane, s^2 its noise.
#[derive(Clone, Debug)]
pub struct PlaneEvidence {
    pub information: DMatrix<f64>,
    pub weighted_residual: DVector<f64>,
    pub weighted_residual_sq: f64,
    pub log_det_noise: f64,
    pub matched: usize,
    pub gated: usize,
}

impl PlaneEvidence {
    pub fn new(columns: usize) -> Self {
        Self {
            information: DMatrix::zeros(columns, columns),
            weighted_residual: DVector::zeros(columns),
            weighted_residual_sq: 0.0,
            log_det_noise: 0.0,
            matched: 0,
            gated: 0,
        }
    }

    /// Adds one point-to-plane row: `residual` the plane distance to remove, `noise` its
    /// variance, `covariance` the block of P over the row's columns.
    pub fn add(
        &mut self,
        row: &DVector<f64>,
        residual: f64,
        noise: f64,
        covariance: &DMatrix<f64>,
        gate: f64,
    ) {
        let predicted = (row.transpose() * covariance * row)[(0, 0)] + noise;
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

/// The point-to-plane row over [position, rotation] and, when the mount is estimated, its
/// correction: d(n . x)/d[position, theta, mount] = [n, -(R [body]x)^T n, -(R E [m]x)^T n],
/// theta the body's right perturbation, mount one about the lidar origin in the body frame.
pub fn plane_row(
    plane: &Plane,
    q: &UnitQuaternion<f64>,
    landed: &Landed,
    with_mount: bool,
) -> DVector<f64> {
    let mut row = DVector::zeros(if with_mount { 9 } else { 6 });
    row.fixed_rows_mut::<3>(0).copy_from(&plane.normal);
    let rotation_row =
        -(q.to_rotation_matrix().matrix() * landed.body.cross_matrix()).transpose() * plane.normal;
    row.fixed_rows_mut::<3>(3).copy_from(&rotation_row);
    if with_mount {
        let mount_row =
            -(landed.rotation * landed.from_origin.cross_matrix()).transpose() * plane.normal;
        row.fixed_rows_mut::<3>(6).copy_from(&mount_row);
    }
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
    fn voxel_planes() -> LidarMapConfig {
        LidarMapConfig {
            matching: Matching::VoxelPlanes,
            ..LidarMapConfig::default()
        }
    }

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
            ..voxel_planes()
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

    /// Two stair treads share a coarse voxel, which is no plane; each has a finer one that is.
    #[test]
    fn a_point_on_a_stair_falls_through_to_its_treads_own_plane() {
        let mut map = PlaneMap::new(voxel_planes());
        for k in 0..64 {
            let (u, v) = ((k % 8) as f64 * 0.03, (k / 8) as f64 * 0.03);
            map.insert(&Vector3::new(0.01 + u, 0.01 + v, 0.10));
            map.insert(&Vector3::new(0.26 + u, 0.01 + v, 0.27));
        }
        let lower = map.plane_at(&Vector3::new(0.1, 0.1, 0.12)).expect("lower tread");
        let upper = map.plane_at(&Vector3::new(0.35, 0.1, 0.3)).expect("upper tread");
        assert!((lower.offset.abs() - 0.10).abs() < 1e-6 && lower.normal.z.abs() > 0.999);
        assert!((upper.offset.abs() - 0.27).abs() < 1e-6 && upper.normal.z.abs() > 0.999);
    }

    fn nearest_points_map() -> PlaneMap {
        PlaneMap::new(LidarMapConfig {
            matching: Matching::NearestPoints,
            point_spacing: 0.02,
            ..LidarMapConfig::default()
        })
    }

    /// The plane through a point's nearest map points, found across cell borders, and none
    /// for a point with no map points within reach.
    #[test]
    fn nearest_points_fit_the_surface_under_a_point_across_cells_and_none_out_of_reach() {
        let mut map = nearest_points_map();
        for i in 0..20 {
            for j in 0..20 {
                // A floor at z = 0.1 straddling the cell border at x = 0.5.
                map.insert(&Vector3::new(0.3 + 0.03 * i as f64, 0.03 * j as f64, 0.1));
            }
        }
        let plane = map.plane_at(&Vector3::new(0.49, 0.2, 0.15)).expect("floor");
        assert!(plane.normal.z.abs() > 0.999);
        assert!((plane.offset.abs() - 0.1).abs() < 1e-6);

        // Farther than a cell from every mapped point: nothing to fit.
        assert!(map.plane_at(&Vector3::new(2.5, 2.5, 2.5)).is_none());
    }

    #[test]
    fn nearest_points_skip_a_point_too_close_to_one_already_mapped() {
        let mut map = nearest_points_map();
        map.insert(&Vector3::new(0.1, 0.1, 0.1));
        map.insert(&Vector3::new(0.11, 0.1, 0.1));
        map.insert(&Vector3::new(0.2, 0.1, 0.1));
        assert_eq!(map.cells.values().map(Vec::len).sum::<usize>(), 2);
    }

    #[test]
    fn a_full_voxel_stops_taking_points() {
        let mut map = PlaneMap::new(LidarMapConfig {
            max_points_per_voxel: 20,
            ..voxel_planes()
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
            ..voxel_planes()
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
        let velocity = Vector3::new(0.5, -0.2, 0.1);
        let rate = Vector3::new(0.3, 1.2, -0.7);
        let origin = Vector3::new(0.1, -0.05, 0.3);
        let mount = UnitQuaternion::from_euler_angles(0.02, -0.03, 0.01);
        let point = SlicePoint {
            body: Vector3::new(3.0, -1.0, 0.4),
            dt: -0.006,
        };
        let distance = |position: &Vector3<f64>, q: &UnitQuaternion<f64>, mount: &UnitQuaternion<f64>| {
            let landed = land(&point, &origin, mount, position, q, &velocity, &rate);
            plane.normal.dot(&landed.world) + plane.offset
        };
        let landed = land(&point, &origin, &mount, &position, &q, &velocity, &rate);
        let row = plane_row(&plane, &q, &landed, true);
        let eps = 1e-6;
        let base = distance(&position, &q, &mount);
        for axis in 0..3 {
            let mut step = Vector3::zeros();
            step[axis] = eps;
            let dp = (distance(&(position + step), &q, &mount) - base) / eps;
            assert!((dp - row[axis]).abs() < 1e-5);
            let turn = UnitQuaternion::from_scaled_axis(step);
            let dq = (distance(&position, &(q * turn), &mount) - base) / eps;
            assert!((dq - row[3 + axis]).abs() < 1e-4, "theta {axis}: {dq} vs {}", row[3 + axis]);
            let dm = (distance(&position, &q, &(turn * mount)) - base) / eps;
            assert!((dm - row[6 + axis]).abs() < 1e-4, "mount {axis}: {dm} vs {}", row[6 + axis]);
        }
    }

    #[test]
    fn the_gate_counts_rather_than_fuses_a_far_point() {
        let mut evidence = PlaneEvidence::new(6);
        let row = DVector::from_vec(vec![0.0, 0.0, 1.0, 0.0, 0.0, 0.0]);
        let p = DMatrix::identity(6, 6) * 1e-4;
        evidence.add(&row, 0.01, 1e-4, &p, 9.0);
        evidence.add(&row, 1.0, 1e-4, &p, 9.0);
        assert_eq!((evidence.matched, evidence.gated), (1, 1));
    }
}
