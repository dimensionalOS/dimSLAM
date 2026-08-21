// Copyright 2026 Dimensional Inc.
// SPDX-License-Identifier: Apache-2.0
//
// Range-gated point cloud from a depth image, in the depth sensor's own frame.

use lcm_msgs::sensor_msgs::{CameraInfo, Image, PointCloud2, PointField};

const BYTES_PER_POINT: i32 = 12;

fn xyz_field(name: &str, offset: i32) -> PointField {
    PointField {
        name: name.to_string(),
        offset,
        datatype: PointField::FLOAT32 as u8,
        count: 1,
    }
}

/// Deprojects every depth pixel through the depth intrinsics and keeps the ones inside the
/// range gate. Stereo depth error grows as range squared, so the far gate is what makes the
/// cloud usable for mapping; `max_range` of 0 leaves it open. Points come out in
/// `depth.header.frame_id`, unrotated, for a consumer that already knows where the sensor is.
pub fn depth_cloud(
    depth: &Image,
    depth_info: &CameraInfo,
    units_per_meter: f64,
    min_range: f64,
    max_range: f64,
    out: &mut PointCloud2,
) {
    let (fx, fy) = (depth_info.K[0], depth_info.K[4]);
    let (cx, cy) = (depth_info.K[2], depth_info.K[5]);
    let far = if max_range > 0.0 { max_range } else { f64::INFINITY };

    out.data.clear();
    let mut points = 0;
    for v in 0..depth.height {
        let row_start = v as usize * depth.step as usize;
        for u in 0..depth.width {
            let pixel_start = row_start + u as usize * 2;
            let raw = u16::from_ne_bytes([depth.data[pixel_start], depth.data[pixel_start + 1]]);
            if raw == 0 {
                continue;
            }
            let z = raw as f64 / units_per_meter;
            if z < min_range || z > far {
                continue;
            }
            let x = (u as f64 - cx) / fx * z;
            let y = (v as f64 - cy) / fy * z;
            out.data.extend_from_slice(&(x as f32).to_le_bytes());
            out.data.extend_from_slice(&(y as f32).to_le_bytes());
            out.data.extend_from_slice(&(z as f32).to_le_bytes());
            points += 1;
        }
    }

    out.header = depth.header.clone();
    out.height = 1;
    out.width = points;
    out.fields = vec![xyz_field("x", 0), xyz_field("y", 4), xyz_field("z", 8)];
    out.is_bigendian = false;
    out.point_step = BYTES_PER_POINT;
    out.row_step = BYTES_PER_POINT * points;
    out.is_dense = true;
}
