// Copyright 2026 Dimensional Inc.
// SPDX-License-Identifier: Apache-2.0
//
// Feeds one recording's camera straight into mono cuVSLAM and prints each frame's raw pose, so a
// tracker that stops moving without reporting itself lost shows up frame by frame.
//
//   cargo run --release --example mono_trace -- <mem2.db> <image stream> <camera_info stream> \
//       [start_s] [duration_s] [distortion_model]

use std::path::PathBuf;

use dim_slam::nalgebra::Isometry3;
use dim_slam::{CameraConfig, CuvslamCore, CuvslamOdometryConfig};
use dim_slam_replay::{db::Recording, lcm};

fn main() -> Result<(), String> {
    tracing_subscriber::fmt().with_writer(std::io::stderr).init();
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 4 {
        return Err("usage: mono_trace <mem2.db> <image stream> <camera_info stream> [start_s] [duration_s] [distortion_model]".into());
    }
    let start_s = args.get(4).map_or(Ok(0.0), |s| s.parse::<f64>()).map_err(|e| e.to_string())?;
    let duration_s = args.get(5).map_or(Ok(0.0), |s| s.parse::<f64>()).map_err(|e| e.to_string())?;
    // camera_info is often published once, at the recording's start, so it is read unwindowed.
    let mut infos = Recording::open(&PathBuf::from(&args[1]), &[args[3].clone()], 0.0, 0.0)?;
    let info_record = infos.next_record()?.ok_or("no camera_info")?;
    let mut info = lcm::camera_info(&info_record.data)?;
    // For a recording that mislabels its lens.
    if let Some(model) = args.get(6) {
        info.distortion_model = model.clone();
    }
    let frame_id = info.frame_id.clone();
    eprintln!("camera_info {frame_id} {}x{} K={:?} {} D={:?}", info.width, info.height, info.intrinsics, info.distortion_model, info.distortion);

    let mut tracker = CuvslamCore::new(CuvslamOdometryConfig {
        camera_mode: "mono".into(),
        cameras: vec![CameraConfig { frame_id: frame_id.clone(), ..Default::default() }],
        output_frame_id: frame_id.clone(),
        ..Default::default()
    })?;
    let identity = |parent: &str, child: &str| (parent == child).then(Isometry3::identity);
    tracker.handle_camera_info(info, &identity);

    let mut images = Recording::open(&PathBuf::from(&args[1]), &[args[2].clone()], start_s, duration_s)?;
    let mut previous: Option<Isometry3<f64>> = None;
    let (mut frames, mut tracked, mut frozen) = (0u64, 0u64, 0u64);
    println!("frame stamp_s tracked frozen angle_since_last_rad translation_since_last cov_x cov_roll");
    while let Some(record) = images.next_record()? {
        let mut image = lcm::image(&record.data)?;
        image.frame_id = frame_id.clone();
        let stamp_s = image.timestamp_ns as f64 / 1e9;
        frames += 1;
        let Some(estimate) = tracker.handle_image(image, &identity) else {
            println!("{frames} {stamp_s:.3} 0 0 nan nan nan nan");
            previous = None;
            continue;
        };
        tracked += 1;
        let pose = estimate.pose;
        let (angle, translation) = previous.map_or((f64::NAN, f64::NAN), |last| {
            let step = last.inverse() * pose;
            (step.rotation.angle(), step.translation.vector.norm())
        });
        // The tracker hands back last frame's pose unchanged: nothing about this frame was solved.
        let is_frozen = previous == Some(pose);
        frozen += u64::from(is_frozen);
        let covariance = estimate.pose_covariance;
        println!(
            "{frames} {stamp_s:.3} 1 {} {angle:.3e} {translation:.3e} {:.4} {:.4}",
            u8::from(is_frozen),
            covariance[(0, 0)],
            covariance[(3, 3)]
        );
        previous = Some(pose);
    }
    eprintln!("frames {frames} tracked {tracked} frozen {frozen} ({:.1}% of tracked)", 100.0 * frozen as f64 / tracked.max(1) as f64);
    Ok(())
}
