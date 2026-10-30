// Copyright 2026 Dimensional Inc.
// SPDX-License-Identifier: Apache-2.0

use nalgebra::{Isometry3, Translation3, UnitQuaternion};

use crate::types::CameraModel;
use cu_vslam_rs::ffi::{CuvPose, CUV_DISTORTION_BROWN, CUV_DISTORTION_PINHOLE};

pub fn to_cuv_pose(iso: &Isometry3<f64>) -> CuvPose {
    let rotation = iso.rotation.quaternion();
    CuvPose {
        rotation_xyzw: [
            rotation.i as f32,
            rotation.j as f32,
            rotation.k as f32,
            rotation.w as f32,
        ],
        translation: [
            iso.translation.x as f32,
            iso.translation.y as f32,
            iso.translation.z as f32,
        ],
    }
}

pub fn cuv_pose_to_isometry(pose: &CuvPose) -> Isometry3<f64> {
    let [x, y, z, w] = pose.rotation_xyzw;
    Isometry3::from_parts(
        Translation3::new(
            pose.translation[0] as f64,
            pose.translation[1] as f64,
            pose.translation[2] as f64,
        ),
        UnitQuaternion::from_quaternion(nalgebra::Quaternion::new(
            w as f64, x as f64, y as f64, z as f64,
        )),
    )
}

/// cuVSLAM's Distortion::Model order; cu_vslam_rs 0.2.0 names only pinhole and Brown.
const CUV_DISTORTION_FISHEYE: u8 = 1;
const CUV_DISTORTION_POLYNOMIAL: u8 = 3;

/// camera_info's distortion as cuVSLAM's. A model or coefficient count cuVSLAM cannot take is an
/// error: tracking through a lens model it was not given bends every ray off the image centre.
pub fn to_distortion(info: &CameraModel) -> Result<(u8, Vec<f32>), String> {
    let d: Vec<f32> = info.distortion.iter().map(|coefficient| *coefficient as f32).collect();
    if d.iter().all(|coefficient| *coefficient == 0.0) {
        return Ok((CUV_DISTORTION_PINHOLE, Vec::new()));
    }
    // Trailing zeros are absent terms (OpenCV pads to 5, 8, 12 or 14).
    let used = d.iter().rposition(|coefficient| *coefficient != 0.0).map_or(0, |last| last + 1);
    match (info.distortion_model.as_str(), used) {
        // ROS orders plumb_bob (k1, k2, p1, p2, k3); four coefficients are OpenCV's k3 = 0.
        // cuVSLAM's Brown wants (k1, k2, k3, p1, p2).
        ("plumb_bob" | "", ..=5) => {
            let term = |index: usize| d.get(index).copied().unwrap_or(0.0);
            Ok((CUV_DISTORTION_BROWN, vec![term(0), term(1), term(4), term(2), term(3)]))
        }
        // kalibr's pinhole-equi and OpenCV's fisheye, which cuVSLAM's Fisheye takes as is.
        ("equidistant" | "fisheye", ..=4) => {
            Ok((CUV_DISTORTION_FISHEYE, (0..4).map(|index| d.get(index).copied().unwrap_or(0.0)).collect()))
        }
        // OpenCV's first eight, which cuVSLAM's Polynomial takes as is.
        ("rational_polynomial" | "plumb_bob" | "", ..=8) => Ok((
            CUV_DISTORTION_POLYNOMIAL,
            (0..8).map(|index| d.get(index).copied().unwrap_or(0.0)).collect(),
        )),
        (model, used) => Err(format!(
            "camera {}: cuVSLAM has no distortion model for {model:?} with {used} coefficients \
             ({:?}); it takes plumb_bob (up to 5), equidistant (4) or rational_polynomial (8)",
            info.frame_id, info.distortion
        )),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn camera(model: &str, distortion: Vec<f64>) -> CameraModel {
        CameraModel {
            distortion_model: model.to_string(),
            distortion,
            ..Default::default()
        }
    }

    #[test]
    fn distortion_all_zero_is_pinhole() {
        let expected = Ok((CUV_DISTORTION_PINHOLE, Vec::new()));
        assert_eq!(to_distortion(&camera("plumb_bob", vec![0.0; 5])), expected);
        assert_eq!(to_distortion(&camera("equidistant", vec![0.0; 4])), expected);
        assert_eq!(to_distortion(&camera("", Vec::new())), expected);
    }

    #[test]
    fn distortion_plumb_bob_reorders_to_brown() {
        let (model, parameters) = to_distortion(&camera("plumb_bob", vec![0.1, 0.2, 0.3, 0.4, 0.5])).unwrap();
        assert_eq!(model, CUV_DISTORTION_BROWN);
        assert_eq!(parameters, vec![0.1, 0.2, 0.5, 0.3, 0.4]); // cuVSLAM: k1 k2 k3 p1 p2
    }

    #[test]
    fn distortion_four_plumb_bob_coefficients_are_kept_with_no_k3() {
        // These used to be dropped as no distortion at all.
        let (model, parameters) = to_distortion(&camera("plumb_bob", vec![0.1, 0.2, 0.3, 0.4])).unwrap();
        assert_eq!(model, CUV_DISTORTION_BROWN);
        assert_eq!(parameters, vec![0.1, 0.2, 0.0, 0.3, 0.4]);
    }

    #[test]
    fn distortion_equidistant_is_fisheye() {
        let go2_front = vec![-0.0731, -0.0234, -0.00693, 0.00924];
        let (model, parameters) = to_distortion(&camera("equidistant", go2_front)).unwrap();
        assert_eq!(model, CUV_DISTORTION_FISHEYE);
        assert_eq!(parameters, vec![-0.0731, -0.0234, -0.00693, 0.00924]);
    }

    #[test]
    fn distortion_rational_polynomial_keeps_opencv_order() {
        let (model, parameters) =
            to_distortion(&camera("rational_polynomial", vec![0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8])).unwrap();
        assert_eq!(model, CUV_DISTORTION_POLYNOMIAL);
        assert_eq!(parameters, vec![0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]);
    }

    #[test]
    fn distortion_cuvslam_cannot_take_is_refused() {
        assert!(to_distortion(&camera("equidistant", vec![0.1, 0.2, 0.3, 0.4, 0.5])).is_err());
        assert!(to_distortion(&camera("double_sphere", vec![0.1])).is_err());
    }

    #[test]
    fn pose_round_trip() {
        let iso = Isometry3::from_parts(
            Translation3::new(1.0, -2.0, 3.0),
            UnitQuaternion::from_euler_angles(0.1, 0.2, 0.3),
        );
        let back = cuv_pose_to_isometry(&to_cuv_pose(&iso));
        assert!((back.translation.vector - iso.translation.vector).norm() < 1.0e-6);
        assert!(back.rotation.angle_to(&iso.rotation) < 1.0e-6);
    }
}
