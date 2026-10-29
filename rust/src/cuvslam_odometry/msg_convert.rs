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

/// ROS orders plumb_bob (k1, k2, p1, p2, k3); cuVSLAM's Brown wants (k1, k2, k3, p1, p2).
pub fn to_distortion(info: &CameraModel) -> (u8, Vec<f32>) {
    let d = &info.distortion;
    let distorted = d.len() >= 5 && d.iter().any(|coefficient| *coefficient != 0.0);
    if !distorted {
        return (CUV_DISTORTION_PINHOLE, Vec::new());
    }
    (
        CUV_DISTORTION_BROWN,
        vec![
            d[0] as f32,
            d[1] as f32,
            d[4] as f32,
            d[2] as f32,
            d[3] as f32,
        ],
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn distortion_all_zero_is_pinhole() {
        let info = CameraModel {
            distortion: vec![0.0; 5],
            ..Default::default()
        };
        assert_eq!(to_distortion(&info), (CUV_DISTORTION_PINHOLE, Vec::new()));
    }

    #[test]
    fn distortion_empty_is_pinhole() {
        let info = CameraModel::default();
        assert_eq!(to_distortion(&info), (CUV_DISTORTION_PINHOLE, Vec::new()));
    }

    #[test]
    fn distortion_plumb_bob_reorders_to_brown() {
        let info = CameraModel {
            distortion: vec![0.1, 0.2, 0.3, 0.4, 0.5],
            ..Default::default()
        };
        let (model, parameters) = to_distortion(&info);
        assert_eq!(model, CUV_DISTORTION_BROWN);
        assert_eq!(parameters, vec![0.1, 0.2, 0.5, 0.3, 0.4]); // cuVSLAM: k1 k2 k3 p1 p2
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
