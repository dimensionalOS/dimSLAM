// Copyright 2026 Dimensional Inc.
// SPDX-License-Identifier: Apache-2.0
//
// The point map Point-LIO matches against, so the two can be compared stage by stage, bit for
// bit: points hashed into cubes, each cube's points kept in the order they arrived, cubes
// dropped least recently touched first past a capacity, and a point's nearest neighbors
// searched in its own cube and the six that share a face with it.

use std::collections::HashMap;

use nalgebra::{Matrix3, Vector3};

use super::scan_points::SurfacePoint;

type Cube = [i32; 3];

/// The cubes searched around a point's own, in the order they are searched.
const FACE_NEIGHBORS: [Cube; 7] = [
    [0, 0, 0],
    [-1, 0, 0],
    [1, 0, 0],
    [0, 1, 0],
    [0, -1, 0],
    [0, 0, -1],
    [0, 0, 1],
];

const NONE: usize = usize::MAX;

struct Cell {
    cube: Cube,
    points: Vec<SurfacePoint>,
    newer: usize,
    older: usize,
}

/// A neighbor candidate. Its size and field order are part of how the partial sorts below
/// break ties, so they match Point-LIO's.
#[derive(Clone, Copy)]
#[repr(C)]
struct Candidate {
    distance_sq: f64,
    cell: u32,
    index: u32,
}

pub struct PointMap {
    inverse_size: f32,
    capacity: usize,
    cubes: HashMap<Cube, usize>,
    cells: Vec<Cell>,
    free: Vec<usize>,
    newest: usize,
    oldest: usize,
    candidates: Vec<Candidate>,
}

impl PointMap {
    /// Cubes of side `cube_size` m; past `capacity` cubes the least recently touched goes.
    pub fn new(cube_size: f32, capacity: usize) -> Self {
        Self {
            inverse_size: (1.0f64 / cube_size as f64) as f32,
            capacity,
            cubes: HashMap::new(),
            cells: Vec::new(),
            free: Vec::new(),
            newest: NONE,
            oldest: NONE,
            candidates: Vec::new(),
        }
    }

    fn cube_of(&self, p: &SurfacePoint) -> Cube {
        p.position.map(|v| (v * self.inverse_size).floor() as i32)
    }

    pub fn cube_count(&self) -> usize {
        self.cubes.len()
    }

    pub fn insert(&mut self, points: &[SurfacePoint]) {
        for &p in points {
            let cube = self.cube_of(&p);
            match self.cubes.get(&cube).copied() {
                Some(cell) => {
                    self.cells[cell].points.push(p);
                    self.unlink(cell);
                    self.make_newest(cell);
                }
                None => {
                    let cell = self.new_cell(cube, p);
                    self.make_newest(cell);
                    self.cubes.insert(cube, cell);
                    if self.cubes.len() >= self.capacity {
                        let oldest = self.oldest;
                        self.unlink(oldest);
                        self.cubes.remove(&self.cells[oldest].cube);
                        self.cells[oldest].points = Vec::new();
                        self.free.push(oldest);
                    }
                }
            }
        }
    }

    /// Adds each of a scan's world points unless the map already holds a point in its cube
    /// of side `spacing`, judged by the neighbors it was matched against (`neighbors[i]` for
    /// `points[i]`; a point matched to none is always added).
    pub fn grow(&mut self, points: &[SurfacePoint], neighbors: &[Vec<SurfacePoint>], spacing: f64) {
        let side = spacing as f32;
        let mut added = Vec::with_capacity(points.len());
        for (p, near) in points.iter().zip(neighbors) {
            if near.is_empty() {
                added.push(*p);
                continue;
            }
            let center = p.position.map(|v| ((v / side).floor() + 0.5) * side);
            let covered = near.iter().any(|q| {
                (0..3).all(|axis| ((q.position[axis] - center[axis]).abs() as f64) < 0.5 * spacing)
            });
            if !covered {
                added.push(*p);
            }
        }
        self.insert(&added);
    }

    /// Up to `count` map points within `range` of `p`, nearest first, then the rest in the
    /// order the partial sorts leave them. False, with `out` untouched, when there are none.
    pub fn nearest(
        &mut self,
        p: &SurfacePoint,
        count: usize,
        range: f64,
        out: &mut Vec<SurfacePoint>,
    ) -> bool {
        let center = self.cube_of(p);
        let mut candidates = std::mem::take(&mut self.candidates);
        candidates.clear();
        for offset in &FACE_NEIGHBORS {
            let cube = [
                center[0] + offset[0],
                center[1] + offset[1],
                center[2] + offset[2],
            ];
            if let Some(&cell) = self.cubes.get(&cube) {
                self.gather(cell, &mut candidates, p, count, range);
            }
        }
        let found = !candidates.is_empty();
        if found {
            if candidates.len() > count {
                candidates.select_nth_unstable_by(count - 1, nearer);
                candidates.truncate(count);
            }
            candidates.select_nth_unstable_by(0, nearer);
            out.clear();
            out.extend(
                candidates
                    .iter()
                    .map(|c| self.cells[c.cell as usize].points[c.index as usize]),
            );
        }
        self.candidates = candidates;
        found
    }

    /// The cell's points within `range`, at most its `count` nearest.
    fn gather(
        &self,
        cell: usize,
        candidates: &mut Vec<Candidate>,
        p: &SurfacePoint,
        count: usize,
        range: f64,
    ) {
        let before = candidates.len();
        for (index, q) in self.cells[cell].points.iter().enumerate() {
            let distance_sq = squared_distance(q, p);
            if distance_sq < range * range {
                candidates.push(Candidate {
                    distance_sq,
                    cell: cell as u32,
                    index: index as u32,
                });
            }
        }
        if before + count < candidates.len() {
            candidates[before..].select_nth_unstable_by(count - 1, nearer);
            candidates.truncate(before + count);
        }
    }

    fn new_cell(&mut self, cube: Cube, p: SurfacePoint) -> usize {
        let cell = Cell {
            cube,
            points: vec![p],
            newer: NONE,
            older: NONE,
        };
        match self.free.pop() {
            Some(slot) => {
                self.cells[slot] = cell;
                slot
            }
            None => {
                self.cells.push(cell);
                self.cells.len() - 1
            }
        }
    }

    fn make_newest(&mut self, cell: usize) {
        self.cells[cell].newer = NONE;
        self.cells[cell].older = self.newest;
        if self.newest != NONE {
            self.cells[self.newest].newer = cell;
        } else {
            self.oldest = cell;
        }
        self.newest = cell;
    }

    fn unlink(&mut self, cell: usize) {
        let (newer, older) = (self.cells[cell].newer, self.cells[cell].older);
        if newer != NONE {
            self.cells[newer].older = older;
        } else {
            self.newest = older;
        }
        if older != NONE {
            self.cells[older].newer = newer;
        } else {
            self.oldest = newer;
        }
    }
}

/// In f32, summed as x^2 + (y^2 + z^2).
fn squared_distance(a: &SurfacePoint, b: &SurfacePoint) -> f64 {
    let d = [0, 1, 2].map(|axis| a.position[axis] - b.position[axis]);
    (d[0] * d[0] + (d[1] * d[1] + d[2] * d[2])) as f64
}

fn nearer(a: &Candidate, b: &Candidate) -> std::cmp::Ordering {
    a.distance_sq.total_cmp(&b.distance_sq)
}

/// A lidar point into the world through the lidar's mount on the body and the body's pose,
/// rounded back to f32; its reflectivity and time carry over.
pub fn to_world(
    p: &SurfacePoint,
    rotation: &Matrix3<f64>,
    position: &Vector3<f64>,
    mount_rotation: &Matrix3<f64>,
    mount_translation: &Vector3<f64>,
) -> SurfacePoint {
    let body = Vector3::new(
        p.position[0] as f64,
        p.position[1] as f64,
        p.position[2] as f64,
    );
    let world = rotation * (mount_rotation * body + mount_translation) + position;
    SurfacePoint {
        position: [world[0] as f32, world[1] as f32, world[2] as f32],
        ..*p
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn at(x: f32, y: f32, z: f32) -> SurfacePoint {
        SurfacePoint {
            position: [x, y, z],
            reflectivity: 0.0,
            time_ms: 0.0,
        }
    }

    #[test]
    fn nearest_searches_the_cube_and_its_faces_within_range() {
        let mut map = PointMap::new(1.0, 1000);
        map.insert(&[
            at(0.5, 0.5, 0.5),
            at(0.9, 0.5, 0.5),
            at(1.5, 0.5, 0.5),  // the +x face neighbor
            at(1.5, 1.5, 0.5),  // an edge neighbor: never searched
            at(-0.5, 0.5, 0.5), // the -x face neighbor, but past the range
        ]);
        let mut out = Vec::new();
        assert!(map.nearest(&at(0.8, 0.5, 0.5), 2, 0.9, &mut out));
        assert_eq!(out.len(), 2);
        assert_eq!(out[0], at(0.9, 0.5, 0.5));
        assert!(out.contains(&at(0.5, 0.5, 0.5)));
        assert!(map.nearest(&at(0.5, 0.5, 0.5), 5, 10.0, &mut out));
        assert_eq!(out.len(), 4);
        assert!(!out.contains(&at(1.5, 1.5, 0.5)));
        let mut untouched = vec![at(9.0, 9.0, 9.0)];
        assert!(!map.nearest(&at(50.0, 0.0, 0.0), 5, 1.0, &mut untouched));
        assert_eq!(untouched, vec![at(9.0, 9.0, 9.0)]);
    }

    #[test]
    fn grow_skips_points_whose_cube_a_neighbor_already_fills() {
        let mut map = PointMap::new(2.0, 1000);
        let points = [at(0.1, 0.1, 0.1), at(0.6, 0.1, 0.1), at(3.1, 0.1, 0.1)];
        let neighbors = vec![vec![at(0.2, 0.3, 0.4)], vec![at(0.2, 0.3, 0.4)], Vec::new()];
        map.grow(&points, &neighbors, 0.5);
        let mut out = Vec::new();
        assert!(!map.nearest(&at(0.1, 0.1, 0.1), 5, 0.2, &mut out));
        assert!(map.nearest(&at(0.6, 0.1, 0.1), 5, 0.01, &mut out));
        assert!(map.nearest(&at(3.1, 0.1, 0.1), 5, 0.01, &mut out));
    }

    #[test]
    fn past_capacity_the_least_recently_touched_cube_goes() {
        let mut map = PointMap::new(1.0, 3);
        map.insert(&[at(0.5, 0.5, 0.5), at(1.5, 0.5, 0.5)]);
        map.insert(&[at(0.6, 0.5, 0.5)]); // touches the first cube again
        map.insert(&[at(5.5, 0.5, 0.5)]); // a third cube: the second goes
        assert_eq!(map.cube_count(), 2);
        let mut out = Vec::new();
        assert!(!map.nearest(&at(1.5, 0.5, 0.5), 5, 0.3, &mut out));
        assert!(map.nearest(&at(0.5, 0.5, 0.5), 5, 0.3, &mut out));
    }
}
