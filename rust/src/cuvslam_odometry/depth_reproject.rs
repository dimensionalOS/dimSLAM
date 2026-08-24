// Copyright 2026 Dimensional Inc.
// SPDX-License-Identifier: Apache-2.0

use dimos_module::nalgebra::Isometry3;
use lcm_msgs::sensor_msgs::{CameraInfo, Image};

/// Nearest surface wins on collision; unhit pixels stay 0, which cuVSLAM reads as no depth.
pub fn reproject_depth(
    depth: &Image,
    depth_info: &CameraInfo,
    camera_info: &CameraInfo,
    camera_from_depth: &Isometry3<f64>,
    units_per_meter: f64,
    out: &mut Image,
) {
    out.header = depth.header.clone();
    out.header.frame_id = camera_info.header.frame_id.clone();
    out.width = camera_info.width;
    out.height = camera_info.height;
    out.encoding = depth.encoding.clone();
    out.is_bigendian = depth.is_bigendian;
    out.step = out.width * std::mem::size_of::<u16>() as i32;
    out.data.clear();
    out.data.resize(out.step as usize * out.height as usize, 0);

    let rotation = camera_from_depth.rotation.to_rotation_matrix();
    let column_x = rotation.matrix().column(0).into_owned();
    let column_y = rotation.matrix().column(1).into_owned();
    let column_z = rotation.matrix().column(2).into_owned();
    let origin = camera_from_depth.translation.vector;
    let (fx_d, fy_d) = (depth_info.K[0], depth_info.K[4]);
    let (cx_d, cy_d) = (depth_info.K[2], depth_info.K[5]);
    let (fx_c, fy_c) = (camera_info.K[0], camera_info.K[4]);
    let (cx_c, cy_c) = (camera_info.K[2], camera_info.K[5]);
    let (out_width, out_height) = (out.width, out.height);

    for v in 0..depth.height {
        let row_start = v as usize * depth.step as usize;
        for u in 0..depth.width {
            let pixel_start = row_start + u as usize * 2;
            let raw = u16::from_ne_bytes([depth.data[pixel_start], depth.data[pixel_start + 1]]);
            if raw == 0 {
                continue;
            }
            let z = raw as f64 / units_per_meter;
            let x = (u as f64 - cx_d) / fx_d * z;
            let y = (v as f64 - cy_d) / fy_d * z;
            let z_c = column_x[2] * x + column_y[2] * y + column_z[2] * z + origin[2];
            if z_c <= 0.0 {
                continue;
            }
            let x_c = column_x[0] * x + column_y[0] * y + column_z[0] * z + origin[0];
            let y_c = column_x[1] * x + column_y[1] * y + column_z[1] * z + origin[1];
            let u_c = (fx_c * x_c / z_c + cx_c).round() as i64;
            let v_c = (fy_c * y_c / z_c + cy_c).round() as i64;
            if u_c < 0 || u_c >= out_width as i64 || v_c < 0 || v_c >= out_height as i64 {
                continue;
            }
            let value = ((z_c * units_per_meter).round() as i64).min(u16::MAX as i64) as u16;
            let slot_start = (v_c as usize * out_width as usize + u_c as usize) * 2;
            let slot = u16::from_ne_bytes([out.data[slot_start], out.data[slot_start + 1]]);
            if slot == 0 || value < slot {
                out.data[slot_start..slot_start + 2].copy_from_slice(&value.to_ne_bytes());
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use dimos_module::nalgebra::{Translation3, UnitQuaternion};

    const UNITS_PER_METER: f64 = 1000.0;

    fn make_info(width: i32, height: i32, focal: f64) -> CameraInfo {
        let mut info = CameraInfo {
            width,
            height,
            ..Default::default()
        };
        info.K[0] = focal;
        info.K[4] = focal;
        info.K[2] = width as f64 / 2.0;
        info.K[5] = height as f64 / 2.0;
        info
    }

    fn make_depth(width: i32, height: i32) -> Image {
        Image {
            width,
            height,
            step: width * 2,
            data: vec![0; (width * height * 2) as usize],
            encoding: "16UC1".into(),
            ..Default::default()
        }
    }

    fn set_pixel(image: &mut Image, u: i32, v: i32, value: u16) {
        let start = (v * image.step + u * 2) as usize;
        image.data[start..start + 2].copy_from_slice(&value.to_ne_bytes());
    }

    fn get_pixel(image: &Image, u: i32, v: i32) -> u16 {
        let start = (v * image.step + u * 2) as usize;
        u16::from_ne_bytes([image.data[start], image.data[start + 1]])
    }

    #[test]
    fn identity_transform_is_passthrough() {
        let info = make_info(8, 6, 100.0);
        let mut depth = make_depth(8, 6);
        set_pixel(&mut depth, 3, 2, 1500);
        let mut out = Image::default();
        reproject_depth(&depth, &info, &info, &Isometry3::identity(), UNITS_PER_METER, &mut out);
        assert_eq!(get_pixel(&out, 3, 2), 1500);
        let total: u64 = (0..6)
            .flat_map(|v| (0..8).map(move |u| (u, v)))
            .map(|(u, v)| get_pixel(&out, u, v) as u64)
            .sum();
        assert_eq!(total, 1500);
    }

    #[test]
    fn translation_shifts_projection() {
        // Moving the target camera +x by 0.1 m at 1 m depth with f=100 shifts u by -10 px.
        let info = make_info(32, 32, 100.0);
        let mut depth = make_depth(32, 32);
        set_pixel(&mut depth, 16, 16, 1000);
        let camera_from_depth =
            Isometry3::from_parts(Translation3::new(-0.1, 0.0, 0.0), UnitQuaternion::identity());
        let mut out = Image::default();
        reproject_depth(&depth, &info, &info, &camera_from_depth, UNITS_PER_METER, &mut out);
        assert_eq!(get_pixel(&out, 6, 16), 1000);
    }

    #[test]
    fn nearest_surface_wins_on_collision() {
        // Two depth pixels, one behind the other, forced onto one target pixel by a
        // low-resolution target camera.
        let depth_info = make_info(4, 4, 1000.0);
        // Principal point at the corner so every ray lands on pixel (0, 0).
        let mut target_info = make_info(1, 1, 0.001);
        target_info.K[2] = 0.0;
        target_info.K[5] = 0.0;
        let mut depth = make_depth(4, 4);
        set_pixel(&mut depth, 1, 1, 3000);
        set_pixel(&mut depth, 2, 2, 1200);
        let mut out = Image::default();
        reproject_depth(
            &depth,
            &depth_info,
            &target_info,
            &Isometry3::identity(),
            UNITS_PER_METER,
            &mut out,
        );
        assert_eq!(get_pixel(&out, 0, 0), 1200);
    }

    #[test]
    fn behind_camera_is_dropped() {
        let info = make_info(8, 8, 100.0);
        let mut depth = make_depth(8, 8);
        set_pixel(&mut depth, 4, 4, 500);
        // Push the point 1 m behind the target camera.
        let camera_from_depth =
            Isometry3::from_parts(Translation3::new(0.0, 0.0, -1.0), UnitQuaternion::identity());
        let mut out = Image::default();
        reproject_depth(&depth, &info, &info, &camera_from_depth, UNITS_PER_METER, &mut out);
        assert!(out.data.iter().all(|byte| *byte == 0));
    }
}
