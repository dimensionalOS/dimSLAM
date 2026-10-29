//! One variant of one scenario through the core, on the recording's own clock: nothing waits,
//! so it runs as fast as the core does.

use std::path::Path;
use std::time::Instant;

use dim_slam::nalgebra::{Isometry3, Quaternion, Translation3, UnitQuaternion, Vector3};
use dim_slam::{
    DimSlamCore, DimSlamCoreConfig, FusionCore, LidarScan, OdometryEstimate, OdometryFusionConfig,
    TfLookup,
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
    /// Each `baselines` stream's poses, in scenario order.
    pub baselines: Vec<Vec<(i64, Isometry3<f64>)>>,
    /// The tag camera in the output frame, when the scenario scores tags.
    pub body_from_camera: Option<Isometry3<f64>>,
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

    fn cloud(&mut self, scan: &LidarScan, tf: &TfBuffer, output_frame: &str) -> Result<(), String> {
        match self {
            Core::Fusion(core) => {
                let mount = tf
                    .latest(output_frame, &scan.frame_id)
                    .ok_or(format!("lidar: tf has no {output_frame} -> {}", scan.frame_id))?;
                core.handle_cloud(scan, &mount);
            }
            Core::Slam(core) => core.handle_cloud(scan, tf),
        }
        Ok(())
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
    Baseline(usize),
    Rest,
    Imu(usize),
    Lidar(usize),
    Image(usize),
    CameraInfo(usize),
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
    msg.timestamp_ns += spec.offset_ns();
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

    // tf first, from the recording's start through the window: mounts are static, often
    // published only once at the start, and every lookup then sees all of them.
    let mut tf = TfBuffer::ignoring(&scenario.tf_ignored_parents);
    let tf_duration = if scenario.duration_seconds > 0.0 {
        scenario.start_seconds + scenario.duration_seconds
    } else {
        0.0
    };
    let mut tf_recording = Recording::open(
        &path,
        std::slice::from_ref(&scenario.tf_stream),
        0.0,
        tf_duration,
    )?;
    while let Some(record) = tf_recording.next_record()? {
        for edge in lcm::tf(&record.data)? {
            tf.insert(edge.parent, edge.child, edge.transform);
        }
    }
    if std::env::var_os("REPLAY_TF_DEBUG").is_some() {
        eprintln!("tf edges: {:?}", tf.edge_names());
    }
    for edge in &scenario.static_tf {
        let [x, y, z, w] = edge.rotation;
        tf.insert(
            edge.parent.clone(),
            edge.child.clone(),
            Isometry3::from_parts(
                Translation3::from(Vector3::from(edge.translation)),
                UnitQuaternion::from_quaternion(Quaternion::new(w, x, y, z)),
            ),
        );
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
    for (index, baseline) in scenario.baselines.iter().enumerate() {
        add(&baseline.stream, Role::Baseline(index));
    }
    add(&scenario.rest.stream, Role::Rest);
    for (index, imu) in scenario.imus.iter().enumerate() {
        add(imu.stream(), Role::Imu(index));
    }
    for (index, lidar) in scenario.lidars.iter().enumerate() {
        add(lidar.stream(), Role::Lidar(index));
    }
    if scenario.mode == Mode::Slam {
        for (index, info) in scenario.camera_infos.iter().enumerate() {
            add(info.stream(), Role::CameraInfo(index));
        }
        for (index, image) in scenario.images.iter().enumerate() {
            add(image.stream(), Role::Image(index));
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
        baselines: vec![Vec::new(); scenario.baselines.len()],
        body_from_camera: scenario
            .tags
            .as_ref()
            .and_then(|spec| tf.latest(&output_frame, &spec.camera_frame)),
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
                Role::Source(index) => {
                    lcm::odometry(&record.data)?.timestamp_ns + scenario.sources[index].offset_ns()
                }
                Role::Reference | Role::Baseline(_) | Role::Rest => 0,
                Role::Imu(index) => {
                    lcm::imu(&record.data)?.timestamp_ns + scenario.imus[index].offset_ns()
                }
                Role::Lidar(index) => {
                    lcm::lidar_stamp(&record.data)? + scenario.lidars[index].offset_ns()
                }
                Role::Image(index) => {
                    (record.ts * 1e9) as i64 + scenario.images[index].offset_ns()
                }
                Role::CameraInfo(index) => {
                    (record.ts * 1e9) as i64 + scenario.camera_infos[index].offset_ns()
                }
            };
            // Publish every tick the data clock has passed, as the live module's timer would.
            if !matches!(role, Role::Reference | Role::Baseline(_) | Role::Rest) {
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
                Role::Baseline(index) => {
                    match remap(&scenario.baselines[index], lcm::odometry(&record.data)?, &tf) {
                        Ok(msg) => run.baselines[index].push((msg.timestamp_ns, msg.pose)),
                        Err(why) => run.dropped.push(why),
                    }
                }
                Role::Rest => {
                    let msg = lcm::odometry(&record.data)?;
                    let still = msg.twist.linear.norm() < scenario.rest.speed
                        && msg.twist.angular.norm() < scenario.rest.angular_speed;
                    let offset = (scenario.rest.time_offset_seconds * 1e9).round() as i64;
                    run.rest.push((msg.timestamp_ns + offset, still));
                }
                Role::Lidar(index) => {
                    let mut scan = lcm::lidar_scan(&record.data)?;
                    scan.timestamp_ns += scenario.lidars[index].offset_ns();
                    if let Some(frame_id) = scenario.lidars[index].frame_id() {
                        scan.frame_id = frame_id.to_string();
                    }
                    if let Err(why) = core.cloud(&scan, &tf, &output_frame) {
                        run.dropped.push(why);
                    }
                }
                Role::Imu(index) => {
                    let mut sample = lcm::imu(&record.data)?;
                    sample.timestamp_ns += scenario.imus[index].offset_ns();
                    if let Some(frame_id) = scenario.imus[index].frame_id() {
                        sample.frame_id = frame_id.to_string();
                    }
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
                Role::Image(index) => {
                    if let Core::Slam(core) = &mut core {
                        let mut image = lcm::image(&record.data)?;
                        image.timestamp_ns += scenario.images[index].offset_ns();
                        if let Some(frame_id) = scenario.images[index].frame_id() {
                            image.frame_id = frame_id.to_string();
                        }
                        core.handle_image(image, &tf);
                    }
                }
                Role::CameraInfo(index) => {
                    if let Core::Slam(core) = &mut core {
                        let mut info = lcm::camera_info(&record.data)?;
                        info.timestamp_ns += scenario.camera_infos[index].offset_ns();
                        if let Some(frame_id) = scenario.camera_infos[index].frame_id() {
                            info.frame_id = frame_id.to_string();
                        }
                        core.handle_camera_info(info, &tf);
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
