//! A scenario: one recording, how its streams feed dimSLAM, the config, named variants of it,
//! and the metrics each variant must meet.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use serde::Deserialize;
use serde_json::Value;

#[derive(Clone, Copy, Debug, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum Mode {
    /// `FusionCore` alone: odometry sources and IMUs, no cameras.
    Fusion,
    /// `DimSlamCore`: cuVSLAM on the images, fused with the sources.
    Slam,
    /// `LioOdometry`: lidar and IMU in Point-LIO's schedule, each scan's pose published.
    PointLio,
}

#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct LioSource {
    /// The frame its poses are in: the source's `parent_frame_id`.
    pub frame_id: String,
    /// g; the accelerometer's clip, when not Point-LIO's 3 g.
    #[serde(default)]
    pub accel_clip_g: Option<f64>,
}

#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct OdometryStream {
    pub stream: String,
    /// Overrides the message's frame_id / child_frame_id, as a blueprint's remapping would.
    #[serde(default)]
    pub frame_id: Option<String>,
    #[serde(default)]
    pub child_frame_id: Option<String>,
    /// Carries the pose through the recording's tf to this frame and makes it the child.
    #[serde(default)]
    pub carry_to: Option<String>,
    /// s added to every stamp: puts a stream recorded on another clock onto the replay's.
    #[serde(default)]
    pub time_offset_seconds: f64,
}

impl OdometryStream {
    pub fn offset_ns(&self) -> i64 {
        (self.time_offset_seconds * 1e9).round() as i64
    }
}

/// parent_from_child, rotation as [x, y, z, w].
#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct StaticTransform {
    pub parent: String,
    pub child: String,
    pub translation: [f64; 3],
    #[serde(default = "identity_rotation")]
    pub rotation: [f64; 4],
}

fn identity_rotation() -> [f64; 4] {
    [0.0, 0.0, 0.0, 1.0]
}

/// A sensor stream, by name alone or with its samples' frame_id overridden.
#[derive(Clone, Debug, Deserialize)]
#[serde(untagged)]
pub enum SensorStream {
    Name(String),
    Spec {
        stream: String,
        #[serde(default)]
        frame_id: Option<String>,
        /// s added to every stamp, as `OdometryStream::time_offset_seconds`.
        #[serde(default)]
        time_offset_seconds: f64,
        /// camera_info only: the lens model, for a recording that mislabels it.
        #[serde(default)]
        distortion_model: Option<String>,
    },
}

impl SensorStream {
    pub fn stream(&self) -> &str {
        match self {
            SensorStream::Name(name) => name,
            SensorStream::Spec { stream, .. } => stream,
        }
    }

    pub fn frame_id(&self) -> Option<&str> {
        match self {
            SensorStream::Name(_) => None,
            SensorStream::Spec { frame_id, .. } => frame_id.as_deref(),
        }
    }

    pub fn distortion_model(&self) -> Option<&str> {
        match self {
            SensorStream::Name(_) => None,
            SensorStream::Spec { distortion_model, .. } => distortion_model.as_deref(),
        }
    }

    pub fn offset_ns(&self) -> i64 {
        match self {
            SensorStream::Name(_) => 0,
            SensorStream::Spec {
                time_offset_seconds,
                ..
            } => (time_offset_seconds * 1e9).round() as i64,
        }
    }
}

/// When the body counts as at rest: this stream's twist under both speeds for at least
/// `min_seconds`; `trim_seconds` comes off each end of a stretch.
#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Rest {
    pub stream: String,
    pub speed: f64,
    pub angular_speed: f64,
    pub min_seconds: f64,
    pub trim_seconds: f64,
    /// s added to the rest stream's stamps, as `OdometryStream::time_offset_seconds`.
    #[serde(default)]
    pub time_offset_seconds: f64,
}

#[derive(Clone, Debug, Default, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Bound {
    pub min: Option<f64>,
    pub max: Option<f64>,
}

#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Scenario {
    pub name: String,
    pub description: String,
    /// Relative to the datasets directory, unless absolute.
    pub recording: String,
    #[serde(default)]
    pub start_seconds: f64,
    /// 0 runs to the end.
    #[serde(default)]
    pub duration_seconds: f64,
    pub mode: Mode,
    pub tf_stream: String,
    /// Mounts the recording's tf lacks, e.g. an IMU's offset inside its lidar from the
    /// sensor's datasheet.
    #[serde(default)]
    pub static_tf: Vec<StaticTransform>,
    /// tf edges under these parents are poses, not mounts, and are left out of lookups.
    #[serde(default)]
    pub tf_ignored_parents: Vec<String>,
    #[serde(default)]
    pub sources: Vec<OdometryStream>,
    #[serde(default)]
    pub imus: Vec<SensorStream>,
    /// PointCloud2 scans fed to the core as lidar measurements.
    #[serde(default)]
    pub lidars: Vec<SensorStream>,
    /// Runs `LioOdometry` on the IMU and lidar streams and hands each of its poses to the core
    /// as an odometry message from this frame (configure it under the config's `sources`); the
    /// lidar then goes to it alone, not to the core.
    #[serde(default)]
    pub lio_source: Option<LioSource>,
    #[serde(default)]
    pub images: Vec<SensorStream>,
    #[serde(default)]
    pub camera_infos: Vec<SensorStream>,
    /// What the output is scored against; never fed in.
    pub reference: OdometryStream,
    /// Other estimators' outputs in the recording, never fed in, scored against the reference
    /// exactly as a variant is (as variant `baseline:<stream>`): what the output has to beat.
    #[serde(default)]
    pub baselines: Vec<OdometryStream>,
    /// Fiducial sightings: every trajectory (output, baselines, reference) is also scored on
    /// how well it puts each tag back where it saw it before. Needs no reference.
    #[serde(default)]
    pub tags: Option<crate::tags::TagSpec>,
    pub rest: Rest,
    /// The core's config: `OdometryFusionConfig` for fusion, `DimSlamCoreConfig` for slam.
    pub config: Value,
    /// Variant name -> JSON pointer -> value set on `config`; null removes the key.
    pub variants: BTreeMap<String, BTreeMap<String, Value>>,
    /// Variant name -> metric -> bound.
    #[serde(default)]
    pub expect: BTreeMap<String, BTreeMap<String, Bound>>,
    /// Informational: what an earlier build scored, which the bounds above were set from.
    #[serde(default)]
    pub baseline: Value,
}

impl Scenario {
    pub fn load(path: &Path) -> Result<Self, String> {
        let text = std::fs::read_to_string(path).map_err(|e| format!("{}: {e}", path.display()))?;
        serde_json::from_str(&text).map_err(|e| format!("{}: {e}", path.display()))
    }

    pub fn recording_path(&self, datasets: &Path) -> PathBuf {
        let path = Path::new(&self.recording);
        if path.is_absolute() {
            path.to_path_buf()
        } else {
            datasets.join(path)
        }
    }

    pub fn variant_config(&self, variant: &str) -> Result<Value, String> {
        let patch = self
            .variants
            .get(variant)
            .ok_or(format!("{}: no variant {variant:?}", self.name))?;
        let mut config = self.config.clone();
        for (pointer, value) in patch {
            set(&mut config, pointer, value.clone())
                .map_err(|e| format!("{}/{variant}: {e}", self.name))?;
        }
        Ok(config)
    }
}

/// RFC 6901 pointer set; the parent must exist. A null value removes the key.
fn set(root: &mut Value, pointer: &str, value: Value) -> Result<(), String> {
    let (parent, key) = pointer
        .rsplit_once('/')
        .ok_or(format!("{pointer:?} is not a JSON pointer"))?;
    let key = key.replace("~1", "/").replace("~0", "~");
    let target = root
        .pointer_mut(parent)
        .ok_or(format!("{pointer:?}: no {parent:?}"))?;
    match target {
        Value::Object(map) if value.is_null() => {
            map.remove(&key);
        }
        Value::Object(map) => {
            map.insert(key, value);
        }
        Value::Array(items) => {
            let index: usize = key
                .parse()
                .map_err(|_| format!("{pointer:?}: not an index"))?;
            *items
                .get_mut(index)
                .ok_or(format!("{pointer:?}: past the end"))? = value;
        }
        _ => {
            return Err(format!(
                "{pointer:?}: {parent:?} is neither object nor array"
            ))
        }
    }
    Ok(())
}
