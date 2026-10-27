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
    /// tf edges under these parents are poses, not mounts, and are left out of lookups.
    #[serde(default)]
    pub tf_ignored_parents: Vec<String>,
    #[serde(default)]
    pub sources: Vec<OdometryStream>,
    #[serde(default)]
    pub imus: Vec<String>,
    #[serde(default)]
    pub images: Vec<String>,
    #[serde(default)]
    pub camera_infos: Vec<String>,
    /// What the output is scored against; never fed in.
    pub reference: OdometryStream,
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
