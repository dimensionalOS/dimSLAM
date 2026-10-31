//! What a run is scored on: how still the output holds at rest, how smooth it moves, and how
//! far it strays from the reference.

use std::collections::BTreeMap;

use dim_slam::nalgebra::{Isometry3, Vector3};

use crate::run::Run;
use crate::scenario::Rest;

pub type Metrics = BTreeMap<String, f64>;

/// The reference posed at `t`: linear in position, slerped in rotation.
pub fn interpolate(track: &[(i64, Isometry3<f64>)], t: i64) -> Option<Isometry3<f64>> {
    let after = track.partition_point(|(ts, _)| *ts < t);
    if after == 0 || after == track.len() {
        return None;
    }
    let (t0, a) = track[after - 1];
    let (t1, b) = track[after];
    let s = if t1 > t0 {
        (t - t0) as f64 / (t1 - t0) as f64
    } else {
        0.0
    };
    Some(a.lerp_slerp(&b, s))
}

/// Stretches the rest stream calls still for long enough, trimmed at both ends, in ns.
pub fn rest_segments(run: &Run, rest: &Rest) -> Vec<(i64, i64)> {
    let min_ns = (rest.min_seconds * 1e9) as i64;
    let trim_ns = (rest.trim_seconds * 1e9) as i64;
    let mut segments = Vec::new();
    let mut start = None;
    let close = |start: i64, end: i64, segments: &mut Vec<(i64, i64)>| {
        if end - start > min_ns {
            segments.push((start + trim_ns, end - trim_ns));
        }
    };
    for (t, still) in &run.rest {
        match (still, start) {
            (true, None) => start = Some(*t),
            (false, Some(s)) => {
                close(s, *t, &mut segments);
                start = None;
            }
            _ => {}
        }
    }
    if let (Some(s), Some((t, _))) = (start, run.rest.last()) {
        close(s, *t, &mut segments);
    }
    segments
}

/// Trajectory error under the yaw and offset that fit the whole run best, so it does not hang
/// on how either frame was oriented at the start: horizontal error per output sample, and the
/// vertical error once the mean height difference is taken out.
pub fn trajectory_errors(run: &Run) -> (Vec<f64>, Vec<f64>) {
    let pairs: Vec<(Vector3<f64>, Vector3<f64>)> = run
        .fused
        .iter()
        .filter_map(|(t, p)| {
            interpolate(&run.reference, *t)
                .map(|r| (p.translation.vector, r.translation.vector))
        })
        .collect();
    if pairs.is_empty() {
        return (Vec::new(), Vec::new());
    }
    let n = pairs.len() as f64;
    let fused_mean: Vector3<f64> = pairs.iter().map(|(f, _)| f).sum::<Vector3<f64>>() / n;
    let reference_mean: Vector3<f64> = pairs.iter().map(|(_, r)| r).sum::<Vector3<f64>>() / n;
    // Closed-form 2D Procrustes: the yaw taking the reference's xy onto the output's.
    let (mut dot, mut cross) = (0.0, 0.0);
    for (f, r) in &pairs {
        let (a, b) = (r - reference_mean, f - fused_mean);
        dot += a.x * b.x + a.y * b.y;
        cross += a.x * b.y - a.y * b.x;
    }
    let angle = cross.atan2(dot);
    let (sin, cos) = angle.sin_cos();
    pairs
        .iter()
        .map(|(f, r)| {
            let a = r - reference_mean;
            let rotated_x = cos * a.x - sin * a.y + fused_mean.x;
            let rotated_y = sin * a.x + cos * a.y + fused_mean.y;
            let horizontal = ((f.x - rotated_x).powi(2) + (f.y - rotated_y).powi(2)).sqrt();
            let vertical = ((f.z - fused_mean.z) - (r.z - reference_mean.z)).abs();
            (horizontal * 1000.0, vertical * 1000.0)
        })
        .unzip()
}

fn yaw(pose: &Isometry3<f64>) -> f64 {
    pose.rotation.euler_angles().2
}

fn rms(values: &[f64]) -> f64 {
    if values.is_empty() {
        f64::NAN
    } else {
        (values.iter().map(|v| v * v).sum::<f64>() / values.len() as f64).sqrt()
    }
}

fn mean(values: &[f64]) -> f64 {
    if values.is_empty() {
        f64::NAN
    } else {
        values.iter().sum::<f64>() / values.len() as f64
    }
}

/// The reference moved into the output's frame at the output's first sample.
pub fn aligned_reference(run: &Run) -> Vec<Option<Isometry3<f64>>> {
    let Some(first) = run
        .fused
        .iter()
        .find(|(t, _)| interpolate(&run.reference, *t).is_some())
    else {
        return vec![None; run.fused.len()];
    };
    let fused_from_reference = first.1
        * interpolate(&run.reference, first.0)
            .expect("found above")
            .inverse();
    run.fused
        .iter()
        .map(|(t, _)| interpolate(&run.reference, *t).map(|r| fused_from_reference * r))
        .collect()
}

pub fn score(run: &Run, segments: &[(i64, i64)]) -> Metrics {
    let at_rest = |t: i64| segments.iter().any(|(a, b)| *a <= t && t <= *b);
    let mut metrics = Metrics::new();
    let (mut peaks_mm, mut peaks_deg) = (Vec::new(), Vec::new());
    for (a, b) in segments {
        let inside: Vec<&Isometry3<f64>> = run
            .fused
            .iter()
            .filter(|(t, _)| a <= t && t <= b)
            .map(|(_, p)| p)
            .collect();
        if inside.len() < 5 {
            continue;
        }
        let extent = |axis: usize| {
            let values = inside.iter().map(|p| p.translation.vector[axis]);
            values.clone().fold(f64::MIN, f64::max) - values.fold(f64::MAX, f64::min)
        };
        peaks_mm.push(extent(0).max(extent(1)) * 1000.0);
        // Unwrapped against the first sample, so a heading near pi is not a full turn.
        let base = yaw(inside[0]);
        let yaws: Vec<f64> = inside
            .iter()
            .map(|p| {
                (yaw(p) - base + std::f64::consts::PI).rem_euclid(std::f64::consts::TAU)
                    - std::f64::consts::PI
            })
            .collect();
        peaks_deg.push(
            (yaws.iter().cloned().fold(f64::MIN, f64::max)
                - yaws.iter().cloned().fold(f64::MAX, f64::min))
            .to_degrees(),
        );
    }
    let xy = |p: &Isometry3<f64>| p.translation.vector.xy();
    // Steps and second differences are taken at the reference's instants (the output's own
    // without one), so outputs published at different cadences are compared alike.
    let sampled: Vec<(i64, Isometry3<f64>)> = if run.reference.is_empty() {
        run.fused.clone()
    } else {
        run.reference
            .iter()
            .filter_map(|(t, _)| interpolate(&run.fused, *t).map(|p| (*t, p)))
            .collect()
    };
    let (mut rest_steps, mut moving_second, mut path) = (Vec::new(), Vec::new(), 0.0);
    for window in sampled.windows(3) {
        let (a, b, c) = (xy(&window[0].1), xy(&window[1].1), xy(&window[2].1));
        let step = (b - a).norm();
        if at_rest(window[1].0) {
            rest_steps.push(step * 1000.0);
        } else {
            moving_second.push((c - 2.0 * b + a).norm() * 1000.0);
            path += step;
        }
    }
    let errors: Vec<f64> = run
        .fused
        .iter()
        .zip(aligned_reference(run))
        .filter_map(|((_, p), r)| r.map(|r| (xy(p) - xy(&r)).norm() * 1000.0))
        .collect();
    let mut sorted = errors.clone();
    sorted.sort_by(f64::total_cmp);
    metrics.insert("rest_p2p_mm_mean".into(), mean(&peaks_mm));
    metrics.insert(
        "rest_p2p_mm_max".into(),
        peaks_mm.iter().cloned().fold(f64::NAN, f64::max),
    );
    metrics.insert("rest_p2p_deg_mean".into(), mean(&peaks_deg));
    metrics.insert("rest_step_rms_mm".into(), rms(&rest_steps));
    metrics.insert("moving_roughness_rms_mm".into(), rms(&moving_second));
    metrics.insert("error_mm_mean".into(), mean(&errors));
    metrics.insert(
        "error_mm_p95".into(),
        sorted
            .get(sorted.len() * 95 / 100)
            .copied()
            .unwrap_or(f64::NAN),
    );
    metrics.insert(
        "error_mm_max".into(),
        sorted.last().copied().unwrap_or(f64::NAN),
    );
    metrics.insert(
        "error_mm_final".into(),
        errors.last().copied().unwrap_or(f64::NAN),
    );
    let (horizontal, vertical) = trajectory_errors(run);
    let mut sorted_horizontal = horizontal.clone();
    sorted_horizontal.sort_by(f64::total_cmp);
    metrics.insert("ate_xy_mm_mean".into(), mean(&horizontal));
    metrics.insert(
        "ate_xy_mm_p95".into(),
        sorted_horizontal
            .get(sorted_horizontal.len() * 95 / 100)
            .copied()
            .unwrap_or(f64::NAN),
    );
    metrics.insert(
        "ate_xy_mm_max".into(),
        sorted_horizontal.last().copied().unwrap_or(f64::NAN),
    );
    metrics.insert("ate_z_mm_mean".into(), mean(&vertical));
    metrics.insert("moving_path_m".into(), path);
    metrics.insert(
        "realtime_factor".into(),
        run.data_seconds / run.wall_seconds.max(1e-9),
    );
    metrics
}

/// The reference scored as though it were the output: what the filter is up against.
pub fn score_reference(run: &Run, segments: &[(i64, i64)]) -> Metrics {
    let as_output = Run {
        variant: "reference".into(),
        fused: run.reference.clone(),
        reference: run.reference.clone(),
        baselines: Vec::new(),
        body_from_camera: None,
        rest: Vec::new(),
        wall_seconds: run.wall_seconds,
        data_seconds: run.data_seconds,
        dropped: Vec::new(),
    };
    let mut metrics = score(&as_output, segments);
    for key in [
        "error_mm_mean",
        "error_mm_p95",
        "error_mm_max",
        "error_mm_final",
        "ate_xy_mm_mean",
        "ate_xy_mm_p95",
        "ate_xy_mm_max",
        "ate_z_mm_mean",
        "realtime_factor",
    ] {
        metrics.remove(key);
    }
    metrics
}
