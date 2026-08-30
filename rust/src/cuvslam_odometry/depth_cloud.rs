// Copyright 2026 Dimensional Inc.
// SPDX-License-Identifier: Apache-2.0
//
// Range-gated point cloud from a depth image, in the depth sensor's own frame.

use crate::types::{CameraModel, ImageFrame, PointCloud};

/// Median, not mean, per block: a flying pixel must not invent a mid-air point.
pub fn depth_cloud(
    depth: &ImageFrame,
    depth_info: &CameraModel,
    units_per_meter: f64,
    min_range: f64,
    max_range: f64,
    decimation: u32,
    out: &mut PointCloud,
) {
    let (fx, fy) = (depth_info.intrinsics[0], depth_info.intrinsics[4]);
    let (cx, cy) = (depth_info.intrinsics[2], depth_info.intrinsics[5]);
    let far = if max_range > 0.0 {
        max_range
    } else {
        f64::INFINITY
    };
    let near_raw = (min_range * units_per_meter).ceil().max(1.0) as u16;
    let far_raw = (far * units_per_meter).min(u16::MAX as f64).floor() as u16;
    let kernel = decimation.max(1) as i32;

    out.points.clear();
    let emit = |u: f64, v: f64, raw: u16, out: &mut PointCloud| {
        let z = raw as f64 / units_per_meter;
        out.points.push([
            ((u - cx) / fx * z) as f32,
            ((v - cy) / fy * z) as f32,
            z as f32,
        ]);
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
        let blocks_w = (depth.width + kernel - 1) / kernel;
        let blocks_h = (depth.height + kernel - 1) / kernel;
        let mut medians: Vec<Option<u16>> = vec![None; (blocks_w * blocks_h) as usize];
        let mut block = Vec::with_capacity((kernel * kernel) as usize);
        for block_v in 0..blocks_h {
            for block_u in 0..blocks_w {
                block.clear();
                for v in block_v * kernel..((block_v + 1) * kernel).min(depth.height) {
                    for u in block_u * kernel..((block_u + 1) * kernel).min(depth.width) {
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
                medians[(block_v * blocks_w + block_u) as usize] = Some(*median);
            }
        }
        let median_at = |block_u: i32, block_v: i32| -> Option<u16> {
            if block_u < 0 || block_v < 0 || block_u >= blocks_w || block_v >= blocks_h {
                return None;
            }
            medians[(block_v * blocks_w + block_u) as usize]
        };
        for block_v in 0..blocks_h {
            for block_u in 0..blocks_w {
                let Some(median) = median_at(block_u, block_v) else {
                    continue;
                };
                // A grazing surface also lies between its neighbours, so only a wider gap is air.
                let margin = (0.3 * units_per_meter).max(0.1 * median as f64) as i32;
                let mid = median as i32;
                let mid_gap = [(1, 0), (0, 1), (1, 1), (1, -1)].iter().any(|&(du, dv)| {
                    let (Some(a), Some(b)) = (
                        median_at(block_u - du, block_v - dv),
                        median_at(block_u + du, block_v + dv),
                    ) else {
                        return false;
                    };
                    let (near, far) = (a.min(b) as i32, a.max(b) as i32);
                    near + margin < mid && mid < far - margin
                });
                if mid_gap {
                    continue;
                }
                emit(
                    (block_u * kernel) as f64 + (kernel - 1) as f64 / 2.0,
                    (block_v * kernel) as f64 + (kernel - 1) as f64 / 2.0,
                    median,
                    out,
                );
            }
        }
    }

    out.timestamp_ns = depth.timestamp_ns;
    out.frame_id.clone_from(&depth.frame_id);
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_info(width: i32, height: i32) -> CameraModel {
        let mut info = CameraModel {
            width,
            height,
            ..Default::default()
        };
        info.intrinsics[0] = 100.0;
        info.intrinsics[4] = 100.0;
        info.intrinsics[2] = width as f64 / 2.0;
        info.intrinsics[5] = height as f64 / 2.0;
        info
    }

    fn make_depth(width: i32, height: i32, millimeters: u16) -> ImageFrame {
        let mut data = Vec::new();
        for _ in 0..width * height {
            data.extend_from_slice(&millimeters.to_ne_bytes());
        }
        ImageFrame {
            width,
            height,
            step: width * 2,
            data,
            encoding: "16UC1".into(),
            ..Default::default()
        }
    }

    fn set_pixel(image: &mut ImageFrame, u: i32, v: i32, millimeters: u16) {
        let start = (v * image.step + u * 2) as usize;
        image.data[start..start + 2].copy_from_slice(&millimeters.to_ne_bytes());
    }

    fn cloud_z_values(cloud: &PointCloud) -> Vec<f32> {
        cloud.points.iter().map(|point| point[2]).collect()
    }

    #[test]
    fn decimation_takes_the_block_median_not_the_mean() {
        let mut depth = make_depth(4, 4, 2000);
        set_pixel(&mut depth, 0, 0, 9000);
        set_pixel(&mut depth, 2, 2, 9000);
        let mut cloud = PointCloud::default();
        depth_cloud(&depth, &make_info(4, 4), 1000.0, 0.0, 0.0, 2, &mut cloud);
        assert_eq!(cloud.points.len(), 4);
        assert!(cloud_z_values(&cloud).iter().all(|&z| z == 2.0));
    }

    #[test]
    fn a_block_with_mostly_invalid_pixels_is_dropped() {
        let mut depth = make_depth(2, 2, 0);
        set_pixel(&mut depth, 0, 0, 2000);
        let mut cloud = PointCloud::default();
        depth_cloud(&depth, &make_info(2, 2), 1000.0, 0.0, 0.0, 2, &mut cloud);
        assert!(cloud.points.is_empty());
    }

    #[test]
    fn decimation_off_emits_every_valid_pixel() {
        let depth = make_depth(4, 4, 2000);
        let mut cloud = PointCloud::default();
        depth_cloud(&depth, &make_info(4, 4), 1000.0, 0.0, 0.0, 1, &mut cloud);
        assert_eq!(cloud.points.len(), 16);
    }

    /// 6x2 image at k=2 = three blocks in a row, each uniform at the given depth.
    fn three_block_strip(left: u16, middle: u16, right: u16) -> PointCloud {
        let mut depth = make_depth(6, 2, left);
        for v in 0..2 {
            for u in 2..4 {
                set_pixel(&mut depth, u, v, middle);
            }
            for u in 4..6 {
                set_pixel(&mut depth, u, v, right);
            }
        }
        let mut cloud = PointCloud::default();
        depth_cloud(&depth, &make_info(6, 2), 1000.0, 0.0, 0.0, 2, &mut cloud);
        cloud
    }

    #[test]
    fn a_block_hanging_mid_gap_between_its_neighbours_is_dropped() {
        // A fringe strip interpolating a wall-to-wall discontinuity: mid-air, no surface.
        let cloud = three_block_strip(1000, 3000, 5000);
        assert_eq!(cloud_z_values(&cloud), vec![1.0, 5.0]);
    }

    #[test]
    fn a_thin_object_in_front_of_the_background_is_kept() {
        // A chair leg: nearer than both neighbours, not between them.
        let cloud = three_block_strip(5000, 1000, 5000);
        assert_eq!(cloud_z_values(&cloud), vec![5.0, 1.0, 5.0]);
    }

    #[test]
    fn a_sloped_surface_within_the_margin_is_kept() {
        // A grazing floor or corridor wall: between its neighbours but inside the margin.
        let cloud = three_block_strip(2000, 2240, 2480);
        assert_eq!(cloud_z_values(&cloud), vec![2.0, 2.24, 2.48]);
    }

    #[test]
    fn range_gate_applies_inside_blocks() {
        let mut depth = make_depth(2, 2, 2000);
        // Two pixels beyond the far gate leave only half the block valid: still kept.
        set_pixel(&mut depth, 0, 0, 7000);
        set_pixel(&mut depth, 1, 0, 7000);
        let mut cloud = PointCloud::default();
        depth_cloud(&depth, &make_info(2, 2), 1000.0, 0.0, 5.0, 2, &mut cloud);
        assert_eq!(cloud.points.len(), 1);
        assert_eq!(cloud_z_values(&cloud), vec![2.0]);
    }
}
