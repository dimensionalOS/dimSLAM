//! Tag revisits: a score that needs no reference trajectory. Each sighting of a fiducial tag
//! is placed in the world through the trajectory being scored; a tag seen on separate visits
//! should land in the same place every time, so the spread between its visits is the drift the
//! trajectory accumulated in between.

use std::collections::BTreeMap;
use std::path::Path;

use dim_slam::nalgebra::{Isometry3, Vector3};
use rusqlite::{Connection, OpenFlags};
use serde::Deserialize;

use crate::lcm;

/// Which detections count, from a PoseStamped stream of tag poses in the camera frame whose
/// row tags carry `marker_id` and, optionally, the detector's quality figures.
#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct TagSpec {
    pub stream: String,
    /// The optical frame the detections are in; placed in the output frame through tf.
    pub camera_frame: String,
    #[serde(default)]
    pub time_offset_seconds: f64,
    /// Sightings farther than this from the camera, m, are too coarse to score drift with.
    pub max_distance_m: f64,
    #[serde(default)]
    pub max_reproj_px: Option<f64>,
    #[serde(default)]
    pub min_tag_px: Option<f64>,
    /// Detector-reported body speeds (its own units), past which motion blur dominates.
    #[serde(default)]
    pub max_lin_speed: Option<f64>,
    #[serde(default)]
    pub max_ang_speed: Option<f64>,
    /// Sightings of a tag this far apart in time belong to separate visits.
    pub visit_gap_seconds: f64,
}

pub struct Sighting {
    pub ts_ns: i64,
    pub marker: i64,
    pub camera_from_tag: Isometry3<f64>,
}

/// Every sighting in the window that passes the spec's gates, in time order.
pub fn load(path: &Path, spec: &TagSpec) -> Result<Vec<Sighting>, String> {
    let connection = Connection::open_with_flags(path, OpenFlags::SQLITE_OPEN_READ_ONLY)
        .map_err(|e| format!("{}: {e}", path.display()))?;
    let quoted = format!("\"{}\"", spec.stream.replace('"', "\"\""));
    let blob = format!("\"{}_blob\"", spec.stream.replace('"', "\"\""));
    let sql = format!(
        "SELECT json(s.tags), b.data FROM {quoted} s JOIN {blob} b ON b.id = s.id ORDER BY s.ts"
    );
    let mut statement = connection.prepare(&sql).map_err(|e| e.to_string())?;
    let rows = statement
        .query_map([], |row| Ok((row.get::<_, String>(0)?, row.get::<_, Vec<u8>>(1)?)))
        .map_err(|e| e.to_string())?;
    let offset_ns = (spec.time_offset_seconds * 1e9).round() as i64;
    let mut sightings = Vec::new();
    for row in rows {
        let (tags, data) = row.map_err(|e| e.to_string())?;
        let tags: serde_json::Value = serde_json::from_str(&tags).map_err(|e| e.to_string())?;
        let Some(marker) = tags.get("marker_id").and_then(|v| v.as_i64()) else {
            continue;
        };
        let number = |key: &str| tags.get(key).and_then(|v| v.as_f64());
        let within = |key: &str, max: Option<f64>| match (max, number(key)) {
            (Some(max), Some(value)) => value <= max,
            _ => true,
        };
        let at_least = |key: &str, min: Option<f64>| match (min, number(key)) {
            (Some(min), Some(value)) => value >= min,
            _ => true,
        };
        if !(within("reproj_px", spec.max_reproj_px)
            && at_least("tag_px", spec.min_tag_px)
            && within("lin_speed", spec.max_lin_speed)
            && within("ang_speed", spec.max_ang_speed))
        {
            continue;
        }
        let (ts_ns, camera_from_tag) = lcm::pose_stamped(&data)?;
        if camera_from_tag.translation.vector.norm() > spec.max_distance_m {
            continue;
        }
        sightings.push(Sighting {
            ts_ns: ts_ns + offset_ns,
            marker,
            camera_from_tag,
        });
    }
    Ok(sightings)
}

/// (mean, max) over revisited tags of the rms distance between a tag's visits and their
/// mean, mm, and how many tags were revisited. A visit's position is the mean of its sightings.
pub fn revisit_error(
    track: &[(i64, Isometry3<f64>)],
    body_from_camera: &Isometry3<f64>,
    sightings: &[Sighting],
    visit_gap_seconds: f64,
    pose_at: impl Fn(&[(i64, Isometry3<f64>)], i64) -> Option<Isometry3<f64>>,
) -> (f64, f64, usize) {
    let gap_ns = (visit_gap_seconds * 1e9) as i64;
    // marker -> visits, each (last stamp, sum of positions, count)
    let mut visits: BTreeMap<i64, Vec<(i64, Vector3<f64>, usize)>> = BTreeMap::new();
    for sighting in sightings {
        let Some(world_from_body) = pose_at(track, sighting.ts_ns) else {
            continue;
        };
        let tag = (world_from_body * body_from_camera * sighting.camera_from_tag)
            .translation
            .vector;
        let marker = visits.entry(sighting.marker).or_default();
        match marker.last_mut() {
            Some((last, sum, count)) if sighting.ts_ns - *last <= gap_ns => {
                *last = sighting.ts_ns;
                *sum += tag;
                *count += 1;
            }
            _ => marker.push((sighting.ts_ns, tag, 1)),
        }
    }
    let errors: Vec<f64> = visits
        .values()
        .filter(|v| v.len() >= 2)
        .map(|v| {
            let positions: Vec<Vector3<f64>> = v.iter().map(|(_, s, n)| s / *n as f64).collect();
            let mean = positions.iter().sum::<Vector3<f64>>() / positions.len() as f64;
            let ms = positions.iter().map(|p| (p - mean).norm_squared()).sum::<f64>()
                / positions.len() as f64;
            ms.sqrt() * 1000.0
        })
        .collect();
    if errors.is_empty() {
        return (f64::NAN, f64::NAN, 0);
    }
    let mean = errors.iter().sum::<f64>() / errors.len() as f64;
    let max = errors.iter().cloned().fold(f64::NAN, f64::max);
    (mean, max, errors.len())
}
