// Copyright 2026 Dimensional Inc.
// SPDX-License-Identifier: Apache-2.0
//
// The plane Point-LIO fits through a point's five map neighbors, to the bit: the least-squares
// solution of a x + b y + c z = -1 by column-pivoting Householder QR in f32, with sums ordered
// as Eigen's fixed-size (SIMD) and dynamic-size reductions order them.

use super::lidar::SurfacePoint;

/// Points each plane is fitted through.
pub const PLANE_POINTS: usize = 5;

/// Lengths of the runs of equal time in a time-ordered scan: each run is one update.
pub fn time_runs(points: &[SurfacePoint]) -> Vec<usize> {
    let mut runs = Vec::with_capacity(points.len());
    let mut length = 0;
    for pair in points.windows(2) {
        length += 1;
        if pair[1].time_ms > pair[0].time_ms {
            runs.push(length);
            length = 0;
        }
    }
    runs.push(length + 1);
    runs
}

/// The unit-normal plane `[a, b, c, d]` (a x + b y + c z + d = 0) through the first
/// `PLANE_POINTS` of `points`, or `None` when any of them lies farther than `tolerance` from it.
pub fn fit_plane(points: &[SurfacePoint], tolerance: f32) -> Option<[f32; 4]> {
    let mut columns = [[0f32; PLANE_POINTS]; 3];
    for (row, p) in points[..PLANE_POINTS].iter().enumerate() {
        for axis in 0..3 {
            columns[axis][row] = p.position[axis];
        }
    }
    let solution = least_squares(&mut columns, [-1.0; PLANE_POINTS]);
    let length = (solution[0] * solution[0]
        + (solution[1] * solution[1] + solution[2] * solution[2]))
        .sqrt();
    let plane = [
        solution[0] / length,
        solution[1] / length,
        solution[2] / length,
        (1.0f64 / length as f64) as f32,
    ];
    let off = |p: &SurfacePoint| {
        (plane[0] * p.position[0] + plane[1] * p.position[1] + plane[2] * p.position[2] + plane[3])
            .abs()
    };
    if points[..PLANE_POINTS].iter().any(|p| off(p) > tolerance) {
        return None;
    }
    Some(plane)
}

fn square(x: f32) -> f32 {
    x * x
}

/// A five-vector's length, summed as one four-wide packet then the last element.
fn length5(v: &[f32; PLANE_POINTS]) -> f32 {
    ((v[0] * v[0] + v[2] * v[2]) + (v[1] * v[1] + v[3] * v[3]) + v[4] * v[4]).sqrt()
}

/// Sum of products: a four-wide packet when there are exactly four, else in order.
fn dot(a: &[f32], b: &[f32]) -> f32 {
    if a.len() == 4 {
        (a[0] * b[0] + a[2] * b[2]) + (a[1] * b[1] + a[3] * b[3])
    } else {
        a.iter().zip(b).fold(0.0, |sum, (&x, &y)| sum + x * y)
    }
}

/// Turns `v` into a reflector: `v[1..]` its essential part; returns (tau, the new v[0]).
fn reflector(v: &mut [f32]) -> (f32, f32) {
    let tail = if v.len() == 1 {
        0.0
    } else {
        dot(&v[1..], &v[1..])
    };
    let head = v[0];
    if tail <= f32::MIN_POSITIVE {
        v[1..].fill(0.0);
        return (0.0, head);
    }
    let mut beta = (square(head) + tail).sqrt();
    if head >= 0.0 {
        beta = -beta;
    }
    for e in &mut v[1..] {
        *e /= head - beta;
    }
    ((beta - head) / beta, beta)
}

/// Applies the reflector (`essential`, `tau`) to `column`.
fn reflect(column: &mut [f32], essential: &[f32], tau: f32) {
    if column.len() == 1 {
        column[0] *= 1.0 - tau;
    } else if tau != 0.0 {
        let mut along = dot(essential, &column[1..]);
        along += column[0];
        column[0] -= tau * along;
        for (c, e) in column[1..].iter_mut().zip(essential) {
            *c -= tau * *e * along;
        }
    }
}

/// argmin |A x - b| for a 5x3 `columns` (overwritten by its factorization).
fn least_squares(columns: &mut [[f32; PLANE_POINTS]; 3], mut b: [f32; PLANE_POINTS]) -> [f32; 3] {
    let eps = f32::EPSILON;
    let mut taus = [0f32; 3];
    let mut swaps = [0usize; 3];
    let mut exact = [0f32; 3];
    let mut running = [0f32; 3];
    for k in 0..3 {
        exact[k] = length5(&columns[k]);
        running[k] = exact[k];
    }
    let largest = running
        .iter()
        .fold(running[0], |m, &v| if m < v { v } else { m });
    let negligible = square(largest * eps) / PLANE_POINTS as f32;
    let recompute_below = eps.sqrt();
    let mut rank = 3;
    for k in 0..3 {
        let (mut pivot, mut best) = (k, running[k]);
        for (j, &r) in running.iter().enumerate().skip(k + 1) {
            if r > best {
                best = r;
                pivot = j;
            }
        }
        if rank == 3 && square(best) < negligible * (PLANE_POINTS - k) as f32 {
            rank = k;
        }
        swaps[k] = pivot;
        if k != pivot {
            columns.swap(k, pivot);
            running.swap(k, pivot);
            exact.swap(k, pivot);
        }
        let (tau, beta) = reflector(&mut columns[k][k..]);
        taus[k] = tau;
        columns[k][k] = beta;
        let mut essential = [0f32; PLANE_POINTS];
        essential[..PLANE_POINTS - k - 1].copy_from_slice(&columns[k][k + 1..]);
        let essential = &essential[..PLANE_POINTS - k - 1];
        for column in columns.iter_mut().skip(k + 1) {
            reflect(&mut column[k..], essential, tau);
        }
        // Shrink the remaining columns' norms by what this row took, recomputing when the
        // shrinking has lost too much precision.
        for j in k + 1..3 {
            if running[j] != 0.0 {
                let mut left = columns[j][k].abs() / running[j];
                left = (1.0 + left) * (1.0 - left);
                if left < 0.0 {
                    left = 0.0;
                }
                if left * square(running[j] / exact[j]) <= recompute_below {
                    exact[j] = dot(&columns[j][k + 1..], &columns[j][k + 1..]).sqrt();
                    running[j] = exact[j];
                } else {
                    running[j] *= left.sqrt();
                }
            }
        }
    }
    let mut order = [0usize, 1, 2];
    for (k, &s) in swaps.iter().enumerate() {
        order.swap(k, s);
    }
    for k in 0..rank {
        reflect(&mut b[k..], &columns[k][k + 1..], taus[k]);
    }
    for i in (0..rank).rev() {
        if b[i] != 0.0 {
            b[i] /= columns[i][i];
            for t in 0..i {
                b[t] -= b[i] * columns[i][t];
            }
        }
    }
    let mut x = [0f32; 3];
    for i in 0..rank {
        x[order[i]] = b[i];
    }
    x
}

#[cfg(test)]
mod tests {
    use super::*;

    fn at(x: f32, y: f32, z: f32, time_ms: f32) -> SurfacePoint {
        SurfacePoint {
            position: [x, y, z],
            reflectivity: 0.0,
            time_ms,
        }
    }

    #[test]
    fn a_plane_through_five_points_on_it_and_none_through_a_bent_five() {
        let z = |x: f32, y: f32| 0.5 * x + 0.25 * y + 1.0;
        let points: Vec<_> = [(0.0, 0.0), (1.0, 0.0), (0.0, 1.0), (1.0, 1.0), (2.0, 0.5)]
            .iter()
            .map(|&(x, y)| at(x, y, z(x, y), 0.0))
            .collect();
        let plane = fit_plane(&points, 0.1).expect("flat");
        let s = 1.0 / (0.5f32 * 0.5 + 0.25 * 0.25 + 1.0).sqrt();
        for (got, want) in plane.iter().zip([0.5 * s, 0.25 * s, -s, s]) {
            assert!((got - want).abs() < 1e-5, "{plane:?}");
        }
        let mut bent = points.clone();
        bent[4].position[2] += 1.0;
        assert!(fit_plane(&bent, 0.1).is_none());
    }

    #[test]
    fn time_runs_split_where_time_increases() {
        let points: Vec<_> = [0.0, 0.0, 1.0, 2.0, 2.0, 2.0]
            .iter()
            .map(|&t| at(0.0, 0.0, 0.0, t))
            .collect();
        assert_eq!(time_runs(&points), vec![2, 1, 3]);
        assert_eq!(time_runs(&[]), vec![1]);
    }
}
