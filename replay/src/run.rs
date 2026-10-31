//! One variant of one scenario through the core, on the recording's own clock: nothing waits,
//! so it runs as fast as the core does.

use std::time::Instant;

use dim_slam::nalgebra::{Isometry3, Quaternion, Translation3, UnitQuaternion, Vector3};
use dim_slam::nalgebra::Matrix3;
use dim_slam::odometry_fusion::lio_odometry::{point_lio_imu, LioConfig, LioOdometry};
use dim_slam::odometry_fusion::{header_seconds, STANDARD_GRAVITY};
use dim_slam::{DimSlamCore, DimSlamCoreConfig, ImuSample, LidarScan, OdometryEstimate, TfLookup};

use crate::db::Recording;
use crate::lcm;
use crate::scenario::{Mode, OdometryStream, Scenario};
use crate::tf::TfBuffer;
use crate::ReplayOptions;

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

/// Both publish as their inputs move them: each pose lands in `Run::fused` as it comes.
enum Core {
    Slam(Box<DimSlamCore>),
    /// Point-LIO alone: the first IMU and the lidar in the output frame, the lidar placed by tf.
    Lio(Box<LioOdometry>, String),
}

impl Core {
    fn source(&mut self, msg: &OdometryEstimate, tf: &TfBuffer) {
        if let Core::Slam(core) = self {
            core.handle_source(msg, tf);
        }
    }

    /// The poses published since the last call, as the module drains after every message.
    fn drain(&mut self, fused: &mut Vec<(i64, Isometry3<f64>)>) {
        if let Core::Slam(core) = self {
            if let Some(msg) = core.maybe_publish() {
                fused.push((msg.timestamp_ns, msg.pose));
            }
        }
        if let Core::Lio(lio, _) = self {
            while let Some(published) = lio.step() {
                if let Some(pose) = published {
                    let [x, y, z, w] = pose.quaternion;
                    fused.push((
                        (pose.stamp * 1e9).round() as i64,
                        Isometry3::from_parts(
                            Translation3::from(pose.position),
                            UnitQuaternion::from_quaternion(Quaternion::new(w, x, y, z)),
                        ),
                    ));
                }
            }
        }
    }

    fn imu(&mut self, sample: &ImuSample) {
        if let Core::Lio(lio, _) = self {
            lio.push_imu(
                0,
                header_seconds(sample.timestamp_ns),
                sample.angular_velocity,
                sample.linear_acceleration / STANDARD_GRAVITY,
            );
        }
    }

    /// `scans_before`: how many scans this run already gave the core.
    fn cloud(
        &mut self,
        scan: &LidarScan,
        tf: &TfBuffer,
        scans_before: u64,
        perturb: Option<(u64, u64)>,
    ) -> Result<(), String> {
        let perturbed = match perturb {
            Some((at, seed)) if scans_before == at => Some(drop_point(scan, seed)),
            _ => None,
        };
        let scan = perturbed.as_ref().unwrap_or(scan);
        match self {
            Core::Slam(core) => core.handle_cloud(scan, tf),
            Core::Lio(lio, output_frame) => {
                if !lio.lidar_mounted(0) {
                    let mount = tf
                        .latest(output_frame, &scan.frame_id)
                        .ok_or(format!("lidar: tf has no {output_frame} -> {}", scan.frame_id))?;
                    lio.set_lidar_mount(
                        0,
                        mount.rotation.to_rotation_matrix().into_inner(),
                        mount.translation.vector,
                    );
                }
                lio.push_scan(0, scan);
            }
        }
        Ok(())
    }

    fn report(&self) {
        if let Core::Slam(core) = self {
            core.report();
        }
    }
}

/// Keeps the points whose index is `k` modulo `n`, with their per-point fields.
fn keep_every(scan: &mut LidarScan, k: usize, n: usize) {
    fn keep<T: Copy>(values: &mut Vec<T>, k: usize, n: usize) {
        *values = values.iter().enumerate().filter(|(i, _)| i % n == k).map(|(_, v)| *v).collect();
    }
    keep(&mut scan.points, k, n);
    keep(&mut scan.offsets_ns, k, n);
    keep(&mut scan.tags, k, n);
    keep(&mut scan.intensities, k, n);
}

fn drop_point(scan: &LidarScan, seed: u64) -> LidarScan {
    let mut scan = scan.clone();
    if !scan.points.is_empty() {
        // Spread over the scan: its first points are often zero-range, which the LIO drops anyway.
        let i = (seed.wrapping_mul(2_654_435_761) as usize) % scan.points.len();
        if scan.offsets_ns.len() > i {
            scan.offsets_ns.remove(i);
        }
        scan.points.remove(i);
        if scan.tags.len() > i {
            scan.tags.remove(i);
        }
        if scan.intensities.len() > i {
            scan.intensities.remove(i);
        }
    }
    scan
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

pub fn run(scenario: &Scenario, variant: &str, options: &ReplayOptions) -> Result<Run, String> {
    let datasets = options.datasets.as_path();
    let config = scenario.variant_config(variant)?;
    let path = scenario.recording_path(datasets);
    let (mut core, output_frame) = match scenario.mode {
        Mode::Slam => {
            let config: DimSlamCoreConfig =
                serde_json::from_value(config).map_err(|e| format!("config: {e}"))?;
            let frame = config.output_frame_id.clone();
            (Core::Slam(Box::new(DimSlamCore::new(config)?)), frame)
        }
        Mode::PointLio => {
            let frame = config
                .get("output_frame_id")
                .and_then(|v| v.as_str())
                .unwrap_or("mid360_imu")
                .to_string();
            let mut imu = point_lio_imu();
            // The sensor's clip levels, when they differ from Point-LIO's defaults.
            if let Some(clip) = config.get("accel_clip_g").and_then(|v| v.as_f64()) {
                imu.clip = imu.clip.map(|(gyro, _)| (gyro, clip));
            }
            if let Some(clip) = config.get("gyro_clip").and_then(|v| v.as_f64()) {
                imu.clip = imu.clip.map(|(_, accel)| (clip, accel));
            }
            let lio = LioConfig::default();
            let walks = [lio.process[3], lio.process[4]];
            let point_variance = lio.lidar.point_variance;
            let mut odometry = LioOdometry::new(lio, vec![(imu, walks)], vec![point_variance]);
            odometry.set_imu_mount(0, Matrix3::identity(), Vector3::zeros());
            (Core::Lio(Box::new(odometry), frame.clone()), frame)
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
    // What the output is scored on is read whole, not windowed: a stream recorded on another
    // clock (a re-stamped offline run, say) would otherwise fall outside the window entirely.
    let mut scoring_names: Vec<String> = Vec::new();
    let mut scoring_roles: Vec<Vec<Role>> = Vec::new();
    let mut add_scoring = |name: &str, role: Role| {
        let index = scoring_names.iter().position(|n| n == name).unwrap_or_else(|| {
            scoring_names.push(name.to_string());
            scoring_roles.push(Vec::new());
            scoring_names.len() - 1
        });
        scoring_roles[index].push(role);
    };
    add_scoring(&scenario.reference.stream, Role::Reference);
    for (index, baseline) in scenario.baselines.iter().enumerate() {
        add_scoring(&baseline.stream, Role::Baseline(index));
    }
    add_scoring(&scenario.rest.stream, Role::Rest);
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
    let mut scoring = Recording::open(&path, &scoring_names, 0.0, 0.0)?;
    while let Some(record) = scoring.next_record()? {
        for role in &scoring_roles[record.stream] {
            let msg = lcm::odometry(&record.data)?;
            match role {
                Role::Reference => match remap(&scenario.reference, msg, &tf) {
                    Ok(msg) => run.reference.push((msg.timestamp_ns, msg.pose)),
                    Err(why) => run.dropped.push(why),
                },
                Role::Baseline(index) => match remap(&scenario.baselines[*index], msg, &tf) {
                    Ok(msg) => run.baselines[*index].push((msg.timestamp_ns, msg.pose)),
                    Err(why) => run.dropped.push(why),
                },
                _ => {
                    let still = msg.twist.linear.norm() < scenario.rest.speed
                        && msg.twist.angular.norm() < scenario.rest.angular_speed;
                    let offset = (scenario.rest.time_offset_seconds * 1e9).round() as i64;
                    run.rest.push((msg.timestamp_ns + offset, still));
                }
            }
        }
    }
    for track in std::iter::once(&mut run.reference).chain(run.baselines.iter_mut()) {
        track.sort_by_key(|(t, _)| *t);
    }
    run.rest.sort_by_key(|(t, _)| *t);
    let (mut first_ts, mut last_ts) = (f64::NAN, f64::NAN);
    let mut turns = TurnCache::open(&scenario.name, variant, !scenario.images.is_empty(), options)?;
    let mut scans_given = 0u64;
    let started = Instant::now();
    while let Some(record) = recording.next_record()? {
        if first_ts.is_nan() {
            first_ts = record.ts;
        }
        last_ts = record.ts;
        for role in roles[record.stream].clone() {
            match role {
                Role::Source(index) => {
                    match remap(&scenario.sources[index], lcm::odometry(&record.data)?, &tf) {
                        Ok(msg) => core.source(&msg, &tf),
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
                    if let Some([k, n]) = scenario.lidars[index].every() {
                        keep_every(&mut scan, k, n);
                    }
                    if let Some(frame) = scenario.lidars[index].points_in() {
                        let to = tf.latest(frame, &scan.frame_id).ok_or(format!(
                            "lidar: tf has no {frame} -> {}",
                            scan.frame_id
                        ))?;
                        for p in &mut scan.points {
                            let moved = to * dim_slam::nalgebra::Point3::new(p[0] as f64, p[1] as f64, p[2] as f64);
                            *p = [moved.x as f32, moved.y as f32, moved.z as f32];
                        }
                        scan.frame_id = frame.to_string();
                    }
                    if let Err(why) = core.cloud(&scan, &tf, scans_given, options.perturb) {
                        run.dropped.push(why);
                    }
                    scans_given += 1;
                }
                Role::Imu(index) => {
                    let mut sample = lcm::imu(&record.data)?;
                    sample.timestamp_ns += scenario.imus[index].offset_ns();
                    if let Some(frame_id) = scenario.imus[index].frame_id() {
                        sample.frame_id = frame_id.to_string();
                    }
                    match &mut core {
                        Core::Slam(core) => core.handle_imu(&sample, &tf),
                        Core::Lio(..) => core.imu(&sample),
                    }
                }
                Role::Image(index) => {
                    if let Core::Slam(core) = &mut core {
                        let (tracked, segment) = match turns.played.as_mut() {
                            Some(played) => played
                                .pop_front()
                                .ok_or("the turn cache has fewer frames than the recording")?,
                            None => {
                                let mut image = lcm::image(&record.data)?;
                                image.timestamp_ns += scenario.images[index].offset_ns();
                                if let Some(frame_id) = scenario.images[index].frame_id() {
                                    image.frame_id = frame_id.to_string();
                                }
                                let (tracked, segment) = core.track_image(image, &tf);
                                if let Some(recorded) = turns.recorded.as_mut() {
                                    recorded.push(turn_line(tracked.as_ref(), segment));
                                }
                                (tracked, segment)
                            }
                        };
                        core.handle_tracked(tracked, segment);
                    }
                }
                Role::CameraInfo(index) => {
                    if let Core::Slam(core) = &mut core {
                        let mut info = lcm::camera_info(&record.data)?;
                        info.timestamp_ns += scenario.camera_infos[index].offset_ns();
                        if let Some(frame_id) = scenario.camera_infos[index].frame_id() {
                            info.frame_id = frame_id.to_string();
                        }
                        if let Some(model) = scenario.camera_infos[index].distortion_model() {
                            info.distortion_model = model.to_string();
                        }
                        core.handle_camera_info(info, &tf);
                    }
                }
            }
            core.drain(&mut run.fused);
        }
    }
    turns.finish()?;
    run.wall_seconds = started.elapsed().as_secs_f64();
    run.data_seconds = last_ts - first_ts;
    core.report();
    run.dropped.dedup();
    Ok(run)
}

/// The tracker's output per image (see `ReplayOptions::turns_record` / `turns_play`).
struct TurnCache {
    played: Option<std::collections::VecDeque<(Option<OdometryEstimate>, u64)>>,
    recorded: Option<Vec<String>>,
    record_to: Option<std::path::PathBuf>,
}

impl TurnCache {
    /// A scenario without images has nothing to cache. Each variant has its own: one may
    /// configure other cameras, or none.
    fn open(scenario: &str, variant: &str, has_images: bool, options: &ReplayOptions) -> Result<Self, String> {
        let file = |dir: &Option<std::path::PathBuf>| {
            dir.as_ref().map(|dir| dir.join(format!("{scenario}.{variant}.jsonl")))
        };
        let played = match file(&options.turns_play).filter(|_| has_images) {
            Some(path) => {
                let text = std::fs::read_to_string(&path)
                    .map_err(|e| format!("turn cache {}: {e}", path.display()))?;
                Some(text.lines().map(parse_turn_line).collect::<Result<_, _>>()?)
            }
            None => None,
        };
        let record_to =
            file(&options.turns_record).filter(|path| has_images && !path.exists());
        let recorded = record_to.as_ref().map(|_| Vec::new());
        Ok(Self { played, recorded, record_to })
    }

    fn finish(self) -> Result<(), String> {
        if let Some(left) = self.played.as_ref().filter(|left| !left.is_empty()) {
            return Err(format!("the turn cache has {} frames past the recording", left.len()));
        }
        if let (Some(path), Some(lines)) = (self.record_to, self.recorded) {
            std::fs::write(&path, lines.join("\n") + "\n")
                .map_err(|e| format!("turn cache {}: {e}", path.display()))?;
        }
        Ok(())
    }
}

/// What the filter reads of a tracked frame: the stamp, the pose, its covariance's diagonal
/// and the segment.
fn turn_line(tracked: Option<&OdometryEstimate>, segment: u64) -> String {
    let estimate = tracked.map(|estimate| {
        let q = estimate.pose.rotation.quaternion();
        let t = estimate.pose.translation.vector;
        let c: Vec<f64> = (0..6).map(|i| estimate.pose_covariance[(i, i)]).collect();
        serde_json::json!([estimate.timestamp_ns, q.w, q.i, q.j, q.k, t.x, t.y, t.z, c])
    });
    serde_json::json!({ "segment": segment, "estimate": estimate }).to_string()
}

fn parse_turn_line(line: &str) -> Result<(Option<OdometryEstimate>, u64), String> {
    let value: serde_json::Value =
        serde_json::from_str(line).map_err(|e| format!("turn cache line: {e}"))?;
    let segment = value["segment"].as_u64().ok_or("turn cache line: no segment")?;
    let estimate = match value["estimate"].as_array() {
        None => None,
        Some(fields) if fields.len() < 9 => {
            return Err("turn cache line: no translation (recorded before cameras were fused \
                        as a whole pose); record it again"
                .into())
        }
        Some(fields) => {
            let at = |i: usize| fields[i].as_f64().unwrap_or(0.0);
            let mut pose_covariance = dim_slam::nalgebra::Matrix6::zeros();
            for (i, c) in fields[8].as_array().into_iter().flatten().enumerate().take(6) {
                pose_covariance[(i, i)] = c.as_f64().unwrap_or(0.0);
            }
            Some(OdometryEstimate {
                timestamp_ns: fields[0].as_i64().unwrap_or(0),
                pose: Isometry3::from_parts(
                    Translation3::new(at(5), at(6), at(7)),
                    UnitQuaternion::new_unchecked(Quaternion::new(at(1), at(2), at(3), at(4))),
                ),
                pose_covariance,
                ..Default::default()
            })
        }
    };
    Ok((estimate, segment))
}
