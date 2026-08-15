// Copyright 2026 Dimensional Inc.
// SPDX-License-Identifier: Apache-2.0

#include <doctest/doctest.h>

#include <cmath>

#include "eskf.hpp"

using eskf::Filter;
using eskf::Mat3;
using eskf::Noise;
using eskf::Vec3;

namespace {

constexpr double GRAVITY = 9.80665;

Filter level_filter() {
    Filter filter;
    filter.noise = Noise{0.01, 0.001, 0.1, 0.01};
    filter.init(Eigen::Quaterniond::Identity(), Vec3::Zero(), 0.1, 0.1, 0.05, 0.01);
    return filter;
}

/// Gravity-compensating accel for a level, stationary body.
const Vec3 STATIC_ACCEL(0.0, 0.0, GRAVITY);

}  // namespace

TEST_CASE("skew is antisymmetric and reproduces the cross product") {
    const Vec3 a(1.0, 2.0, 3.0);
    const Vec3 b(-0.5, 4.0, 0.25);
    const Mat3 a_skew = eskf::skew(a);
    CHECK((a_skew + a_skew.transpose()).norm() < 1e-12);
    CHECK((a_skew * b - a.cross(b)).norm() < 1e-12);
}

TEST_CASE("exp and log of so3 round trip") {
    const Vec3 rotation(0.3, -0.2, 0.9);
    CHECK((eskf::log_so3(eskf::exp_so3(rotation)) - rotation).norm() < 1e-9);
}

TEST_CASE("exp of a tiny rotation stays finite and normalized") {
    const Eigen::Quaterniond q = eskf::exp_so3(Vec3(1e-14, 0.0, 0.0));
    CHECK(std::abs(q.norm() - 1.0) < 1e-12);
}

TEST_CASE("stationary propagation holds the state") {
    Filter filter = level_filter();
    for (int i = 0; i < 100; ++i) {
        filter.propagate(0.01, Vec3::Zero(), STATIC_ACCEL);
    }
    CHECK(filter.x.p.norm() < 1e-9);
    CHECK(filter.x.v.norm() < 1e-9);
}

TEST_CASE("propagation integrates constant velocity") {
    Filter filter = level_filter();
    filter.x.v = Vec3(1.0, 0.0, 0.0);
    for (int i = 0; i < 100; ++i) {
        filter.propagate(0.01, Vec3::Zero(), STATIC_ACCEL);
    }
    CHECK(std::abs(filter.x.p.x() - 1.0) < 1e-9);
}

TEST_CASE("propagation grows uncertainty") {
    Filter filter = level_filter();
    const double before = filter.P.trace();
    for (int i = 0; i < 100; ++i) {
        filter.propagate(0.01, Vec3::Zero(), STATIC_ACCEL);
    }
    CHECK(filter.P.trace() > before);
}

TEST_CASE("a position update pulls the estimate") {
    Filter filter = level_filter();
    filter.propagate(0.01, Vec3::Zero(), STATIC_ACCEL);
    Eigen::VectorXd residual(3);
    residual << 1.0, 0.0, 0.0;
    Eigen::MatrixXd jacobian = Eigen::MatrixXd::Zero(3, 15);
    jacobian.block<3, 3>(0, 0).setIdentity();
    Eigen::VectorXd variance = Eigen::VectorXd::Constant(3, 1e-4);
    CHECK(filter.update(residual, jacobian, variance, 0.0));
    CHECK(filter.x.p.x() > 0.9);
    CHECK(filter.P.trace() < 15.0);
}

TEST_CASE("position updates correct velocity through the cross covariance") {
    Filter filter = level_filter();
    filter.x.v = Vec3(1.0, 0.0, 0.0);  // true velocity is zero; the filter is wrong
    for (int step = 0; step < 50; ++step) {
        filter.propagate(0.01, Vec3::Zero(), STATIC_ACCEL);
        Eigen::VectorXd residual(3);
        // The truth stays at the origin, so the residual is minus the estimate.
        residual << -filter.x.p.x(), -filter.x.p.y(), -filter.x.p.z();
        Eigen::MatrixXd jacobian = Eigen::MatrixXd::Zero(3, 15);
        jacobian.block<3, 3>(0, 0).setIdentity();
        Eigen::VectorXd variance = Eigen::VectorXd::Constant(3, 1e-6);
        filter.update(residual, jacobian, variance, 0.0);
    }
    CHECK(std::abs(filter.x.v.x()) < 0.1);
}

TEST_CASE("the mahalanobis gate rejects an outlier and passes an inlier") {
    Filter filter = level_filter();
    filter.propagate(0.01, Vec3::Zero(), STATIC_ACCEL);
    Eigen::MatrixXd jacobian = Eigen::MatrixXd::Zero(3, 15);
    jacobian.block<3, 3>(0, 0).setIdentity();
    Eigen::VectorXd variance = Eigen::VectorXd::Constant(3, 1e-4);
    Eigen::VectorXd outlier(3);
    outlier << 100.0, 0.0, 0.0;
    CHECK_FALSE(filter.update(outlier, jacobian, variance, 3.0));
    Eigen::VectorXd inlier(3);
    inlier << 0.01, 0.0, 0.0;
    CHECK(filter.update(inlier, jacobian, variance, 3.0));
}

TEST_CASE("a gate of zero disables rejection") {
    Filter filter = level_filter();
    Eigen::MatrixXd jacobian = Eigen::MatrixXd::Zero(3, 15);
    jacobian.block<3, 3>(0, 0).setIdentity();
    Eigen::VectorXd variance = Eigen::VectorXd::Constant(3, 1e-4);
    Eigen::VectorXd outlier(3);
    outlier << 100.0, 0.0, 0.0;
    CHECK(filter.update(outlier, jacobian, variance, 0.0));
}

TEST_CASE("a gyro bias propagates into a tilt the accel residual can expose") {
    Filter filter = level_filter();
    const Vec3 true_bias(0.02, 0.0, 0.0);
    // The gyro reads only its bias while the body is still: the estimate tilts.
    for (int i = 0; i < 100; ++i) {
        filter.propagate(0.01, true_bias, STATIC_ACCEL);
    }
    const Vec3 tilt = eskf::log_so3(filter.x.q);
    CHECK(std::abs(tilt.x() - 0.02) < 1e-3);
}

TEST_CASE("an angular rate update observes the gyro bias") {
    Filter filter = level_filter();
    const Vec3 gyro_reading(0.05, 0.0, 0.0);  // still body, biased gyro
    for (int step = 0; step < 50; ++step) {
        filter.propagate(0.01, gyro_reading, STATIC_ACCEL);
        // A perfect rate sensor says the body is not rotating: residual is
        // 0 - (gyro - bg), jacobian -I on the bias block.
        Eigen::VectorXd residual(3);
        const Vec3 predicted = gyro_reading - filter.x.bg;
        residual << -predicted.x(), -predicted.y(), -predicted.z();
        Eigen::MatrixXd jacobian = Eigen::MatrixXd::Zero(3, 15);
        jacobian.block<3, 3>(0, 9) = -Mat3::Identity();
        Eigen::VectorXd variance = Eigen::VectorXd::Constant(3, 1e-6);
        filter.update(residual, jacobian, variance, 0.0);
    }
    CHECK(std::abs(filter.x.bg.x() - 0.05) < 1e-3);
}

TEST_CASE("init seeds the covariance from the stds") {
    Filter filter;
    filter.init(Eigen::Quaterniond::Identity(), Vec3(0.1, 0.0, 0.0), 2.0, 3.0, 0.5, 0.01);
    CHECK(std::abs(filter.P(0, 0) - 4.0) < 1e-12);
    CHECK(std::abs(filter.P(3, 3) - 9.0) < 1e-12);
    CHECK(std::abs(filter.P(6, 6) - 0.25) < 1e-12);
    CHECK(std::abs(filter.P(9, 9) - 1e-4) < 1e-12);
    CHECK(std::abs(filter.x.bg.x() - 0.1) < 1e-12);
}
