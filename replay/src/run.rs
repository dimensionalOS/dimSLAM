//! One variant of one scenario through the core, on the recording's own clock: nothing waits,
//! so it runs as fast as the core does.

use std::path::Path;
use std::time::Instant;

use dim_slam::nalgebra::Isometry3;
use dim_slam::{
    DimSlamCore, DimSlamCoreConfig, FusionCore, OdometryEstimate, OdometryFusionConfig, TfLookup,
};

use crate::db::Recording;
use crate::lcm;
use crate::scenario::{Mode, OdometryStream, Scenario};
use crate::tf::TfBuffer;

pub struct Run {
    pub variant: String,
    /// The core's output at each publish tick, ns.
    pub fused: Vec<(i64, Isometry3<f64>)>,
    pub reference: Vec<(i64, Isometry3<f64>)>,
    /// The rest stream's samples: stamp and whether the body was still.
    pub rest: Vec<(i64, bool)>,
    pub wall_seconds: f64,
    pub data_seconds: f64,
    /// Messages that could not be used, with why.
    pub dropped: Vec<String>,
}

enum Core {
    Fusion(Box<FusionCore>),
    Slam(Box<DimSlamCore>),
}

impl Core {
    fn source(&mut self, msg: &OdometryEstimate) {
        match self {
            Core::Fusion(core) => core.handle_source(msg),
            Core::Slam(core) => core.handle_source(msg),
        }
    }

    fn publish(&mut self, now_ns: i64) -> Option<OdometryEstimate> {
        match self {
            Core::Fusion(core) => core.maybe_publish_at(now_ns),
            Core::Slam(core) => core.maybe_publish_at(now_ns),
        }
    }

    fn report(&self) {
        match self {
            Core::Fusion(core) => core.report(),
            Core::Slam(core) => core.report(),
        }
    }
}

#[derive(Clone, Copy)]
enum Role {
    Source(usize),
    Reference,
    Rest,
    Imu,
    Image,
    CameraInfo,
}

/// Applies a stream's frame overrides and carries the pose to its `carry_to` frame.
fn remap(
    spec: &OdometryStream,
    mut msg: OdometryEstimate,
    tf: &TfBuffer,
) -> Result<OdometryEstimate, String> {
    if let Some(to) = &spec.carry_to {
        let child_from_to = tf.latest(&msg.child_frame_id, to).ok_or(format!(
            "{}: tf has no {} -> {to}",
            spec.stream, msg.child_frame_id
        ))?;
        msg.pose *= child_from_to;
        msg.child_frame_id = to.clone();
    }
    if let Some(frame_id) = &spec.frame_id {
        msg.frame_id = frame_id.clone();
    }
    if let Some(child) = &spec.child_frame_id {
        msg.child_frame_id = child.clone();
    }
    Ok(msg)
}

pub fn run(scenario: &Scenario, variant: &str, datasets: &Path) -> Result<Run, String> {
    let config = scenario.variant_config(variant)?;
    let path = scenario.recording_path(datasets);
    let (mut core, publish_rate, output_frame) = match scenario.mode {
        Mode::Fusion => {
            let config: OdometryFusionConfig =
                serde_json::from_value(config).map_err(|e| format!("config: {e}"))?;
            let (rate, frame) = (config.publish_rate, config.output_frame_id.clone());
            (Core::Fusion(Box::new(FusionCore::new(config))), rate, frame)
        }
        Mode::Slam => {
            let config: DimSlamCoreConfig =
                serde_json::from_value(config).map_err(|e| format!("config: {e}"))?;
            let (rate, frame) = (config.publish_rate, config.output_frame_id.clone());
            (Core::Slam(Box::new(DimSlamCore::new(config)?)), rate, frame)
        }
    };

    // tf first, whole window: mounts are static, and every lookup then sees all of them.
    let mut tf = TfBuffer::ignoring(&scenario.tf_ignored_parents);
    let mut tf_recording = Recording::open(
        &path,
        std::slice::from_ref(&scenario.tf_stream),
        scenario.start_seconds,
        scenario.duration_seconds,
    )?;
    while let Some(record) = tf_recording.next_record()? {
        for edge in lcm::tf(&record.data)? {
            tf.insert(edge.parent, edge.child, edge.transform);
        }
    }

    let mut names: Vec<String> = Vec::new();
    let mut roles: Vec<Vec<Role>> = Vec::new();
    let mut add = |name: &str, role: Role| {
        let index = names.iter().position(|n| n == name).unwrap_or_else(|| {
            names.push(name.to_string());
            roles.push(Vec::new());
            names.len() - 1
        });
        roles[index].push(role);
    };
    for (index, source) in scenario.sources.iter().enumerate() {
        add(&source.stream, Role::Source(index));
    }
    add(&scenario.reference.stream, Role::Reference);
    add(&scenario.rest.stream, Role::Rest);
    for name in &scenario.imus {
        add(name, Role::Imu);
    }
    if scenario.mode == Mode::Slam {
        for name in &scenario.camera_infos {
            add(name, Role::CameraInfo);
        }
        for name in &scenario.images {
            add(name, Role::Image);
        }
    }
    let mut recording = Recording::open(
        &path,
        &names,
        scenario.start_seconds,
        scenario.duration_seconds,
    )?;

    let period_ns = (1.0e9 / publish_rate) as i64;
    let mut run = Run {
        variant: variant.to_string(),
        fused: Vec::new(),
        reference: Vec::new(),
        rest: Vec::new(),
        wall_seconds: 0.0,
        data_seconds: 0.0,
        dropped: Vec::new(),
    };
    let mut next_tick: Option<i64> = None;
    let (mut first_ts, mut last_ts) = (f64::NAN, f64::NAN);
    let started = Instant::now();
    while let Some(record) = recording.next_record()? {
        if first_ts.is_nan() {
            first_ts = record.ts;
        }
        last_ts = record.ts;
        for role in roles[record.stream].clone() {
            let stamp = match role {
                Role::Source(_) | Role::Reference | Role::Rest => {
                    lcm::odometry(&record.data)?.timestamp_ns
                }
                Role::Imu => lcm::imu(&record.data)?.timestamp_ns,
                Role::Image | Role::CameraInfo => (record.ts * 1e9) as i64,
            };
            // Publish every tick the data clock has passed, as the live module's timer would.
            if !matches!(role, Role::Reference | Role::Rest) {
                let tick = next_tick.get_or_insert(stamp);
                while *tick <= stamp {
                    if let Some(msg) = core.publish(*tick) {
                        run.fused.push((*tick, msg.pose));
                    }
                    *tick += period_ns;
                }
            }
            match role {
                Role::Source(index) => {
                    match remap(&scenario.sources[index], lcm::odometry(&record.data)?, &tf) {
                        Ok(msg) => core.source(&msg),
                        Err(why) => run.dropped.push(why),
                    }
                }
                Role::Reference => {
                    match remap(&scenario.reference, lcm::odometry(&record.data)?, &tf) {
                        Ok(msg) => run.reference.push((msg.timestamp_ns, msg.pose)),
                        Err(why) => run.dropped.push(why),
                    }
                }
                Role::Rest => {
                    let msg = lcm::odometry(&record.data)?;
                    let still = msg.twist.linear.norm() < scenario.rest.speed
                        && msg.twist.angular.norm() < scenario.rest.angular_speed;
                    run.rest.push((msg.timestamp_ns, still));
                }
                Role::Imu => {
                    let sample = lcm::imu(&record.data)?;
                    match &mut core {
                        Core::Fusion(core) => match tf.latest(&output_frame, &sample.frame_id) {
                            Some(mount) => core.handle_imu(&sample, &mount),
                            None => run.dropped.push(format!(
                                "imu: tf has no {output_frame} -> {}",
                                sample.frame_id
                            )),
                        },
                        Core::Slam(core) => core.handle_imu(&sample, &tf),
                    }
                }
                Role::Image => {
                    if let Core::Slam(core) = &mut core {
                        core.handle_image(lcm::image(&record.data)?, &tf);
                    }
                }
                Role::CameraInfo => {
                    if let Core::Slam(core) = &mut core {
                        core.handle_camera_info(lcm::camera_info(&record.data)?, &tf);
                    }
                }
            }
        }
    }
    run.wall_seconds = started.elapsed().as_secs_f64();
    run.data_seconds = last_ts - first_ts;
    core.report();
    run.dropped.dedup();
    Ok(run)
}
