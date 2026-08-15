// Copyright 2026 Dimensional Inc.
// SPDX-License-Identifier: Apache-2.0
//
// Error-state Kalman filter for odometry fusion. The IMU is the process model:
// it propagates position, velocity, orientation, and both biases, while the
// covariance carries the cross-terms that let position-space measurements
// correct velocity and bias. Measurements arrive as stacked scalar rows with a
// diagonal noise, so callers pick dimensions freely.
//
// Error-state order: p(0:3) v(3:6) theta(6:9) bg(9:12) ba(12:15), with theta a
// right (body-frame) rotation perturbation: q_true = q_est * Exp(theta).

#pragma once

#include <Eigen/Dense>
#include <Eigen/Geometry>

#include <cmath>

namespace eskf {

using Vec3 = Eigen::Vector3d;
using Mat3 = Eigen::Matrix3d;
using Mat15 = Eigen::Matrix<double, 15, 15>;
using Vec15 = Eigen::Matrix<double, 15, 1>;

inline Mat3 skew(const Vec3& v) {
    Mat3 out;
    out << 0.0, -v.z(), v.y(),
           v.z(), 0.0, -v.x(),
           -v.y(), v.x(), 0.0;
    return out;
}

inline Eigen::Quaterniond exp_so3(const Vec3& rotation_vector) {
    const double angle = rotation_vector.norm();
    if (angle < 1e-12) {
        return Eigen::Quaterniond(1.0, 0.5 * rotation_vector.x(), 0.5 * rotation_vector.y(),
                                  0.5 * rotation_vector.z())
            .normalized();
    }
    return Eigen::Quaterniond(Eigen::AngleAxisd(angle, rotation_vector / angle));
}

inline Vec3 log_so3(const Eigen::Quaterniond& q) {
    const Eigen::AngleAxisd aa(q.normalized());
    return aa.angle() * aa.axis();
}

struct Noise {
    double gyro_noise_density{0.0};   ///< rad/s/sqrt(Hz)
    double gyro_random_walk{0.0};     ///< rad/s^2/sqrt(Hz)
    double accel_noise_density{0.0};  ///< m/s^2/sqrt(Hz)
    double accel_random_walk{0.0};    ///< m/s^3/sqrt(Hz)
};

struct State {
    Vec3 p = Vec3::Zero();  ///< world position of the body
    Vec3 v = Vec3::Zero();  ///< world velocity
    Eigen::Quaterniond q = Eigen::Quaterniond::Identity();  ///< world_from_body
    Vec3 bg = Vec3::Zero();  ///< gyro bias
    Vec3 ba = Vec3::Zero();  ///< accel bias
};

class Filter {
public:
    Noise noise{};
    double gravity{9.80665};

    State x;
    Mat15 P = Mat15::Identity();

    void init(const Eigen::Quaterniond& world_from_body, const Vec3& gyro_bias,
              double position_std, double velocity_std, double rotation_std, double bias_std) {
        x = State{};
        x.q = world_from_body.normalized();
        x.bg = gyro_bias;
        P.setZero();
        P.block<3, 3>(0, 0) = Mat3::Identity() * position_std * position_std;
        P.block<3, 3>(3, 3) = Mat3::Identity() * velocity_std * velocity_std;
        P.block<3, 3>(6, 6) = Mat3::Identity() * rotation_std * rotation_std;
        P.block<3, 3>(9, 9) = Mat3::Identity() * bias_std * bias_std;
        P.block<3, 3>(12, 12) = Mat3::Identity() * bias_std * bias_std;
    }

    void propagate(double dt, const Vec3& gyro, const Vec3& accel) {
        const Mat3 R = x.q.toRotationMatrix();
        const Vec3 unbiased_gyro = gyro - x.bg;
        const Vec3 unbiased_accel = accel - x.ba;
        const Vec3 gravity_vector(0.0, 0.0, -gravity);
        const Vec3 world_accel = R * unbiased_accel + gravity_vector;

        Mat15 F = Mat15::Identity();
        F.block<3, 3>(0, 3) = Mat3::Identity() * dt;
        F.block<3, 3>(3, 6) = -R * skew(unbiased_accel) * dt;
        F.block<3, 3>(3, 12) = -R * dt;
        F.block<3, 3>(6, 6) = Mat3::Identity() - skew(unbiased_gyro) * dt;
        F.block<3, 3>(6, 9) = -Mat3::Identity() * dt;

        x.p += x.v * dt + 0.5 * world_accel * dt * dt;
        x.v += world_accel * dt;
        x.q = (x.q * exp_so3(unbiased_gyro * dt)).normalized();

        Mat15 Q = Mat15::Zero();
        const double gyro_var = noise.gyro_noise_density * noise.gyro_noise_density * dt;
        const double accel_var = noise.accel_noise_density * noise.accel_noise_density * dt;
        Q.block<3, 3>(3, 3) = Mat3::Identity() * accel_var;
        Q.block<3, 3>(6, 6) = Mat3::Identity() * gyro_var;
        Q.block<3, 3>(9, 9) =
            Mat3::Identity() * noise.gyro_random_walk * noise.gyro_random_walk * dt;
        Q.block<3, 3>(12, 12) =
            Mat3::Identity() * noise.accel_random_walk * noise.accel_random_walk * dt;

        P = F * P * F.transpose() + Q;
    }

    /// One stacked update with a diagonal noise. `gate` is a Mahalanobis
    /// threshold in standard deviations per degree of freedom; 0 disables the
    /// gate. Returns false when the measurement was rejected.
    bool update(const Eigen::VectorXd& residual, const Eigen::MatrixXd& H,
                const Eigen::VectorXd& variance, double gate) {
        const Eigen::MatrixXd S =
            H * P * H.transpose() + Eigen::MatrixXd(variance.asDiagonal());
        const Eigen::MatrixXd S_inverse = S.inverse();
        if (gate > 0.0) {
            const double mahalanobis_sq = residual.dot(S_inverse * residual);
            if (!(mahalanobis_sq < gate * gate * static_cast<double>(residual.size()))) {
                return false;
            }
        }
        const Eigen::MatrixXd K = P * H.transpose() * S_inverse;
        const Vec15 dx = K * residual;
        const Mat15 IKH = Mat15::Identity() - K * H;
        P = IKH * P * IKH.transpose() +
            K * Eigen::MatrixXd(variance.asDiagonal()) * K.transpose();
        inject(dx);
        return true;
    }

private:
    void inject(const Vec15& dx) {
        x.p += dx.segment<3>(0);
        x.v += dx.segment<3>(3);
        x.q = (x.q * exp_so3(dx.segment<3>(6))).normalized();
        x.bg += dx.segment<3>(9);
        x.ba += dx.segment<3>(12);
    }
};

}  // namespace eskf
