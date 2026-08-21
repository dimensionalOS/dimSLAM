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

/// Deprojects depth pixels through the depth intrinsics and keeps the ones inside the
/// range gate. Stereo depth error grows as range squared, so the far gate is what makes the
/// cloud usable for mapping; `max_range` of 0 leaves it open. Points come out in
/// `depth.header.frame_id`, unrotated, for a consumer that already knows where the sensor is.
///
/// `decimation` <= 1 emits every pixel. Otherwise each k x k block becomes at most one
/// point: the median of its in-gate depths, deprojected at the block centre. Median rather
/// than mean because averaging across a depth discontinuity invents a point mid-air between
/// foreground and background (the classic flying pixel); the median snaps to one surface,
/// which is why the RealSense D400 post-processing decimation filter uses it. Blocks where
/// fewer than half the pixels have valid in-gate depth are dropped outright — those sit on
/// object edges or specular holes where stereo depth is least trustworthy.
pub fn depth_cloud(
    depth: &Image,
    depth_info: &CameraInfo,
    units_per_meter: f64,
    min_range: f64,
    max_range: f64,
    decimation: u32,
    out: &mut PointCloud2,
) {
    let (fx, fy) = (depth_info.K[0], depth_info.K[4]);
    let (cx, cy) = (depth_info.K[2], depth_info.K[5]);
    let far = if max_range > 0.0 { max_range } else { f64::INFINITY };
    let near_raw = (min_range * units_per_meter).ceil().max(1.0) as u16;
    let far_raw = (far * units_per_meter).min(u16::MAX as f64).floor() as u16;
    let kernel = decimation.max(1) as i32;

    out.data.clear();
    let mut points = 0;
    let mut emit = |u: f64, v: f64, raw: u16, out: &mut PointCloud2| {
        let z = raw as f64 / units_per_meter;
        out.data.extend_from_slice(&(((u - cx) / fx * z) as f32).to_le_bytes());
        out.data.extend_from_slice(&(((v - cy) / fy * z) as f32).to_le_bytes());
        out.data.extend_from_slice(&(z as f32).to_le_bytes());
        points += 1;
    };
    let raw_at = |u: i32, v: i32| {
        let pixel_start = v as usize * depth.step as usize + u as usize * 2;
        u16::from_ne_bytes([depth.data[pixel_start], depth.data[pixel_start + 1]])
    };

    if kernel == 1 {
        for v in 0..depth.height {
            for u in 0..depth.width {
                let raw = raw_at(u, v);
                if raw >= near_raw && raw <= far_raw {
                    emit(u as f64, v as f64, raw, out);
                }
            }
        }
    } else {
        let mut block = Vec::with_capacity((kernel * kernel) as usize);
        for block_v in (0..depth.height).step_by(kernel as usize) {
            for block_u in (0..depth.width).step_by(kernel as usize) {
                block.clear();
                for v in block_v..(block_v + kernel).min(depth.height) {
                    for u in block_u..(block_u + kernel).min(depth.width) {
                        let raw = raw_at(u, v);
                        if raw >= near_raw && raw <= far_raw {
                            block.push(raw);
                        }
                    }
                }
                if block.len() < (kernel * kernel) as usize / 2 {
                    continue;
                }
                let mid = block.len() / 2;
                let (_, median, _) = block.select_nth_unstable(mid);
                let median = *median;
                emit(
                    block_u as f64 + (kernel - 1) as f64 / 2.0,
                    block_v as f64 + (kernel - 1) as f64 / 2.0,
                    median,
                    out,
                );
            }
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

#[cfg(test)]
mod tests {
    use super::*;

    fn make_info(width: i32, height: i32) -> CameraInfo {
        let mut info = CameraInfo {
            width,
            height,
            ..Default::default()
        };
        info.K[0] = 100.0;
        info.K[4] = 100.0;
        info.K[2] = width as f64 / 2.0;
        info.K[5] = height as f64 / 2.0;
        info
    }

    fn make_depth(width: i32, height: i32, millimeters: u16) -> Image {
        let mut data = Vec::new();
        for _ in 0..width * height {
            data.extend_from_slice(&millimeters.to_ne_bytes());
        }
        Image {
            width,
            height,
            step: width * 2,
            data,
            encoding: "16UC1".into(),
            ..Default::default()
        }
    }

    fn set_pixel(image: &mut Image, u: i32, v: i32, millimeters: u16) {
        let start = (v * image.step + u * 2) as usize;
        image.data[start..start + 2].copy_from_slice(&millimeters.to_ne_bytes());
    }

    fn cloud_z_values(cloud: &PointCloud2) -> Vec<f32> {
        cloud
            .data
            .chunks(BYTES_PER_POINT as usize)
            .map(|point| f32::from_le_bytes([point[8], point[9], point[10], point[11]]))
            .collect()
    }

    #[test]
    fn decimation_takes_the_block_median_not_the_mean() {
        let mut depth = make_depth(4, 4, 2000);
        // One flying pixel per block; a mean would land mid-air, the median stays on the wall.
        set_pixel(&mut depth, 0, 0, 9000);
        set_pixel(&mut depth, 2, 2, 9000);
        let mut cloud = PointCloud2::default();
        depth_cloud(&depth, &make_info(4, 4), 1000.0, 0.0, 0.0, 2, &mut cloud);
        assert_eq!(cloud.width, 4);
        assert!(cloud_z_values(&cloud).iter().all(|&z| z == 2.0));
    }

    #[test]
    fn a_block_with_mostly_invalid_pixels_is_dropped() {
        let mut depth = make_depth(2, 2, 0);
        set_pixel(&mut depth, 0, 0, 2000);
        let mut cloud = PointCloud2::default();
        depth_cloud(&depth, &make_info(2, 2), 1000.0, 0.0, 0.0, 2, &mut cloud);
        assert_eq!(cloud.width, 0);
    }

    #[test]
    fn decimation_off_emits_every_valid_pixel() {
        let depth = make_depth(4, 4, 2000);
        let mut cloud = PointCloud2::default();
        depth_cloud(&depth, &make_info(4, 4), 1000.0, 0.0, 0.0, 1, &mut cloud);
        assert_eq!(cloud.width, 16);
    }

    #[test]
    fn range_gate_applies_inside_blocks() {
        let mut depth = make_depth(2, 2, 2000);
        // Two pixels beyond the far gate leave only half the block valid: still kept.
        set_pixel(&mut depth, 0, 0, 7000);
        set_pixel(&mut depth, 1, 0, 7000);
        let mut cloud = PointCloud2::default();
        depth_cloud(&depth, &make_info(2, 2), 1000.0, 0.0, 5.0, 2, &mut cloud);
        assert_eq!(cloud.width, 1);
        assert_eq!(cloud_z_values(&cloud), vec![2.0]);
    }
}
