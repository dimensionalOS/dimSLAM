// Copyright 2026 Dimensional Inc.
// SPDX-License-Identifier: Apache-2.0
//
// Error-state Kalman filter fusing IMU propagation with any number of
// nav_msgs/Odometry sources, told apart by header.frame_id.
//
// in:  imu, odometry
// out: odometry, tf
//
// Source semantics come from the frame ids alone: a source whose header.frame_id
// equals odom_frame is fused absolutely; any other frame drifts on its own, so
// consecutive poses are fused as filter-anchored deltas. Twist is fused in the
// body frame either way. Per-source, per-dimension variances pick the trust:
// negative uses the message covariance, zero drops the dimension, positive is a
// fixed variance. Late measurements roll the filter back and replay.

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <deque>
#include <optional>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

#include <Eigen/Dense>
#include <Eigen/Geometry>

#include "dimos/native.hpp"
#include "nav_msgs/Odometry.hpp"
#include "sensor_msgs/Imu.hpp"
#include "tf2_msgs/TFMessage.hpp"

#include "utils/eskf.hpp"

using dimos::native::Builder;
using dimos::native::Config;
using dimos::native::Module;
using dimos::native::Output;
namespace logging = dimos::native::log;
using Transform = Eigen::Isometry3d;
using eskf::Vec3;
using eskf::Vec15;
using eskf::Mat15;

namespace {

constexpr std::int64_t NS_PER_SEC = 1000000000LL;

std::int64_t stamp_to_ns(const std_msgs::Header& header) {
    return static_cast<std::int64_t>(header.stamp.sec) * NS_PER_SEC +
           static_cast<std::int64_t>(header.stamp.nsec);
}

std_msgs::Time to_stamp(std::int64_t timestamp_ns) {
    std_msgs::Time stamp{};
    stamp.sec = static_cast<std::int32_t>(timestamp_ns / NS_PER_SEC);
    stamp.nsec = static_cast<std::int32_t>(timestamp_ns % NS_PER_SEC);
    return stamp;
}

}  // namespace

struct FusedOdometryConfig {
    std::string odom_frame;
    std::string base_frame;
    /// Output cadence; the filter itself runs at the IMU rate.
    double publish_rate;
    /// How far back a late measurement can reach before it is dropped.
    double replay_buffer_seconds;
    /// Outlier gate in standard deviations per measurement dimension; 0 disables.
    double mahalanobis_gate;
    double imu_gyro_noise_density;   ///< rad/s/sqrt(Hz)
    double imu_gyro_random_walk;     ///< rad/s^2/sqrt(Hz)
    double imu_accel_noise_density;  ///< m/s^2/sqrt(Hz)
    double imu_accel_random_walk;    ///< m/s^3/sqrt(Hz)
    double gravity;
    /// Samples averaged while stationary to level the filter and take the gyro bias.
    double imu_init_samples;
    double initial_position_std;
    double initial_velocity_std;
    double initial_rotation_std;
    double initial_bias_std;
    /// One entry per source: the header.frame_id its messages carry.
    std::vector<std::string> source_frames;
    /// 6 per source, [x y z roll pitch yaw]: <0 message covariance, 0 drop, >0 fixed.
    /// For a drifting (non-odom_frame) source a fixed variance is usually the right
    /// choice, since its message covariance describes accumulated drift, not the delta.
    std::vector<double> source_pose_variances;
    /// 6 per source, [vx vy vz wx wy wz], same convention, body frame.
    std::vector<double> source_twist_variances;
    /// Virtual zero-twist measurement [vx vy vz wx wy wz] applied with every source
    /// message: >0 constrains that dimension toward zero with this variance (e.g. the
    /// non-holonomic vy, vz of a differential-drive base), 0 leaves it free.
    std::vector<double> constraint_twist_variances;
};

class FusedOdometry : public Module {
public:
    void build(Builder& builder, Config& config) override {
        cfg_ = config.parse<FusedOdometryConfig>();
        const std::size_t sources = cfg_.source_frames.size();
        if (cfg_.source_pose_variances.size() != 6 * sources ||
            cfg_.source_twist_variances.size() != 6 * sources) {
            throw std::runtime_error(
                "source_pose_variances and source_twist_variances need 6 entries per "
                "source_frames entry");
        }
        if (cfg_.constraint_twist_variances.size() != 6) {
            throw std::runtime_error("constraint_twist_variances needs exactly 6 entries");
        }
        filter_.noise = eskf::Noise{cfg_.imu_gyro_noise_density, cfg_.imu_gyro_random_walk,
                                    cfg_.imu_accel_noise_density, cfg_.imu_accel_random_walk};
        filter_.gravity = cfg_.gravity;
        anchors_.assign(sources, Transform::Identity());
        anchored_.assign(sources, false);

        builder.input<sensor_msgs::Imu>("imu", &FusedOdometry::on_imu, this);
        builder.input<nav_msgs::Odometry>("odometry", &FusedOdometry::on_odometry, this);
        odometry_ = builder.output<nav_msgs::Odometry>("odometry");
        tf_ = builder.output<tf2_msgs::TFMessage>("tf");
    }

    void teardown() override {
        logging::info("fused_odometry shutting down",
                      {logging::Field("imu_samples", static_cast<std::int64_t>(imu_samples_)),
                       logging::Field("measurements", static_cast<std::int64_t>(measurements_)),
                       logging::Field("gated", static_cast<std::int64_t>(gated_)),
                       logging::Field("replayed", static_cast<std::int64_t>(replayed_)),
                       logging::Field("too_late", static_cast<std::int64_t>(too_late_))});
    }

private:
    struct Measurement {
        std::size_t source{0};
        bool absolute{false};
        bool has_pose{false};
        Transform pose = Transform::Identity();  ///< absolute pose, or the anchored target
        Transform delta = Transform::Identity();  ///< consecutive-message delta when drifting
        std::array<double, 6> pose_variance{};    ///< resolved; <=0 means dropped
        std::array<double, 6> twist_variance{};
        Vec3 linear = Vec3::Zero();
        Vec3 angular = Vec3::Zero();
    };

    /// One filter step, with the state it left behind so a late arrival can roll
    /// back to just before its own slot and replay everything after it.
    struct Event {
        std::int64_t ts_ns{0};
        bool is_imu{false};
        Vec3 gyro = Vec3::Zero();
        Vec3 accel = Vec3::Zero();
        Measurement measurement{};
        // snapshot after processing
        eskf::State state{};
        Mat15 covariance = Mat15::Identity();
        std::vector<Transform> anchors;
        std::vector<bool> anchored;
        std::int64_t last_imu_ns{0};
        Vec3 last_gyro = Vec3::Zero();
    };

    void on_imu(const sensor_msgs::Imu& imu) {
        const Vec3 gyro(imu.angular_velocity.x, imu.angular_velocity.y, imu.angular_velocity.z);
        const Vec3 accel(imu.linear_acceleration.x, imu.linear_acceleration.y,
                         imu.linear_acceleration.z);
        if (!initialized_) {
            init_gyro_sum_ += gyro;
            init_accel_sum_ += accel;
            ++init_samples_;
            if (init_samples_ >= static_cast<std::uint64_t>(cfg_.imu_init_samples)) {
                initialize(stamp_to_ns(imu.header));
            }
            return;
        }
        Event event{};
        event.ts_ns = stamp_to_ns(imu.header);
        event.is_imu = true;
        event.gyro = gyro;
        event.accel = accel;
        insert(std::move(event));
        ++imu_samples_;
        maybe_publish();
    }

    void on_odometry(const nav_msgs::Odometry& msg) {
        if (!initialized_) {
            return;
        }
        const auto found = std::find(cfg_.source_frames.begin(), cfg_.source_frames.end(),
                                     msg.header.frame_id);
        if (found == cfg_.source_frames.end()) {
            DIMOS_LOG_THROTTLED(logging::Level::Warn, logging::from_secs(10),
                                ("odometry from unconfigured frame_id '" + msg.header.frame_id +
                                 "' dropped")
                                    .c_str());
            return;
        }
        const std::size_t source =
            static_cast<std::size_t>(found - cfg_.source_frames.begin());

        Measurement measurement{};
        measurement.source = source;
        measurement.absolute = msg.header.frame_id == cfg_.odom_frame;
        for (int dim = 0; dim < 6; ++dim) {
            measurement.pose_variance[dim] = resolve_variance(
                cfg_.source_pose_variances[source * 6 + dim], msg.pose.covariance[dim * 7]);
            measurement.twist_variance[dim] = resolve_variance(
                cfg_.source_twist_variances[source * 6 + dim], msg.twist.covariance[dim * 7]);
        }
        const Eigen::Quaterniond orientation(msg.pose.pose.orientation.w,
                                             msg.pose.pose.orientation.x,
                                             msg.pose.pose.orientation.y,
                                             msg.pose.pose.orientation.z);
        const bool pose_valid = std::abs(orientation.norm() - 1.0) < 0.1;
        const bool pose_wanted =
            std::any_of(measurement.pose_variance.begin(), measurement.pose_variance.end(),
                        [](double v) { return v > 0.0; });
        if (pose_wanted && pose_valid) {
            Transform pose = Transform::Identity();
            pose.linear() = orientation.normalized().toRotationMatrix();
            pose.translation() = Vec3(msg.pose.pose.position.x, msg.pose.pose.position.y,
                                      msg.pose.pose.position.z);
            if (measurement.absolute) {
                measurement.has_pose = true;
                measurement.pose = pose;
            } else {
                std::optional<Transform>& previous = previous_source_pose_[source];
                if (previous.has_value()) {
                    measurement.has_pose = true;
                    measurement.delta = previous->inverse() * pose;
                }
                previous = pose;
            }
        }
        measurement.linear = Vec3(msg.twist.twist.linear.x, msg.twist.twist.linear.y,
                                  msg.twist.twist.linear.z);
        measurement.angular = Vec3(msg.twist.twist.angular.x, msg.twist.twist.angular.y,
                                   msg.twist.twist.angular.z);

        Event event{};
        event.ts_ns = stamp_to_ns(msg.header);
        event.is_imu = false;
        event.measurement = measurement;
        insert(std::move(event));
        ++measurements_;
        maybe_publish();
    }

    /// Fixed and message policies share a guard: a non-finite or non-positive
    /// variance can only drop the dimension, never divide by it.
    static double resolve_variance(double configured, double from_message) {
        const double variance = configured < 0.0 ? from_message : configured;
        return std::isfinite(variance) && variance > 0.0 ? variance : 0.0;
    }

    void initialize(std::int64_t timestamp_ns) {
        const Vec3 mean_accel = init_accel_sum_ / static_cast<double>(init_samples_);
        const Vec3 mean_gyro = init_gyro_sum_ / static_cast<double>(init_samples_);
        // Stationary accel reads the specific force R^T (0,0,g): level so it maps to +z.
        const Eigen::Quaterniond world_from_body =
            Eigen::Quaterniond::FromTwoVectors(mean_accel, Vec3::UnitZ());
        filter_.init(world_from_body, mean_gyro, cfg_.initial_position_std,
                     cfg_.initial_velocity_std, cfg_.initial_rotation_std,
                     cfg_.initial_bias_std);
        initialized_ = true;
        last_publish_ns_ = timestamp_ns;
        Event seed{};
        seed.ts_ns = timestamp_ns;
        seed.is_imu = true;
        snapshot(seed);
        events_.push_back(std::move(seed));
        logging::info("fused_odometry initialized",
                      {logging::Field("gyro_bias", mean_gyro.norm()),
                       logging::Field("accel_norm", mean_accel.norm())});
    }

    void insert(Event&& event) {
        if (event.ts_ns >= events_.back().ts_ns) {
            process(event, events_.back());
            events_.push_back(std::move(event));
        } else if (event.ts_ns <= events_.front().ts_ns) {
            ++too_late_;
            DIMOS_LOG_THROTTLED(logging::Level::Warn, logging::from_secs(10),
                                "measurement older than the replay buffer dropped");
            return;
        } else {
            // Roll back to the event just before the late slot and replay forward.
            auto slot = std::upper_bound(
                events_.begin(), events_.end(), event.ts_ns,
                [](std::int64_t ts, const Event& e) { return ts < e.ts_ns; });
            slot = events_.insert(slot, std::move(event));
            for (auto it = slot; it != events_.end(); ++it) {
                process(*it, *std::prev(it));
            }
            ++replayed_;
        }
        const std::int64_t horizon =
            events_.back().ts_ns -
            static_cast<std::int64_t>(cfg_.replay_buffer_seconds * 1.0e9);
        while (events_.size() > 1 && events_.front().ts_ns < horizon) {
            events_.pop_front();
        }
    }

    /// Runs one event against the filter restored from `previous`'s snapshot,
    /// then snapshots into `event`. Replay is just calling this again in order.
    void process(Event& event, const Event& previous) {
        filter_.x = previous.state;
        filter_.P = previous.covariance;
        anchors_ = previous.anchors;
        anchored_.assign(previous.anchored.begin(), previous.anchored.end());
        last_imu_ns_ = previous.last_imu_ns;
        last_gyro_ = previous.last_gyro;

        if (event.is_imu) {
            const double dt = static_cast<double>(event.ts_ns - last_imu_ns_) / 1.0e9;
            if (dt > 0.0 && dt < 1.0) {
                filter_.propagate(dt, event.gyro, event.accel);
            }
            last_imu_ns_ = event.ts_ns;
            last_gyro_ = event.gyro;
        } else {
            apply(event.measurement);
        }
        snapshot(event);
    }

    void apply(const Measurement& measurement) {
        std::vector<double> residuals;
        std::vector<double> variances;
        std::vector<Vec15> rows;

        if (measurement.has_pose) {
            Transform target = measurement.pose;
            if (!measurement.absolute) {
                if (!anchored_[measurement.source]) {
                    // First delta after (re)start: adopt the filter pose, fuse nothing.
                    anchor(measurement.source);
                    return;
                }
                target = anchors_[measurement.source] * measurement.delta;
            }
            const Vec3 position_residual = target.translation() - filter_.x.p;
            const Vec3 rotation_residual = eskf::log_so3(
                filter_.x.q.conjugate() * Eigen::Quaterniond(target.linear()));
            for (int dim = 0; dim < 3; ++dim) {
                if (measurement.pose_variance[dim] > 0.0) {
                    Vec15 row = Vec15::Zero();
                    row(dim) = 1.0;
                    rows.push_back(row);
                    residuals.push_back(position_residual(dim));
                    variances.push_back(measurement.pose_variance[dim]);
                }
            }
            for (int dim = 0; dim < 3; ++dim) {
                if (measurement.pose_variance[3 + dim] > 0.0) {
                    Vec15 row = Vec15::Zero();
                    row(6 + dim) = 1.0;
                    rows.push_back(row);
                    residuals.push_back(rotation_residual(dim));
                    variances.push_back(measurement.pose_variance[3 + dim]);
                }
            }
        }

        add_twist_rows(measurement.linear, measurement.angular, measurement.twist_variance,
                       rows, residuals, variances);
        const bool accepted = update(rows, residuals, variances);
        if (accepted && measurement.has_pose && !measurement.absolute) {
            anchor(measurement.source);
        }

        // The configured constraints ride along with every source message.
        std::vector<double> constraint_residuals;
        std::vector<double> constraint_variances;
        std::vector<Vec15> constraint_rows;
        std::array<double, 6> constraint_variance{};
        std::copy(cfg_.constraint_twist_variances.begin(),
                  cfg_.constraint_twist_variances.end(), constraint_variance.begin());
        add_twist_rows(Vec3::Zero(), Vec3::Zero(), constraint_variance, constraint_rows,
                       constraint_residuals, constraint_variances);
        update(constraint_rows, constraint_residuals, constraint_variances);
    }

    /// Body-frame twist rows. Linear velocity observes v and theta; angular velocity
    /// is measured against the latest gyro sample, so it observes the gyro bias.
    void add_twist_rows(const Vec3& linear, const Vec3& angular,
                        const std::array<double, 6>& variance, std::vector<Vec15>& rows,
                        std::vector<double>& residuals, std::vector<double>& variances) {
        const Eigen::Matrix3d body_from_world = filter_.x.q.conjugate().toRotationMatrix();
        const Vec3 body_velocity = body_from_world * filter_.x.v;
        const Eigen::Matrix3d velocity_skew = eskf::skew(body_velocity);
        for (int dim = 0; dim < 3; ++dim) {
            if (variance[dim] > 0.0) {
                Vec15 row = Vec15::Zero();
                row.segment<3>(3) = body_from_world.row(dim);
                row.segment<3>(6) = velocity_skew.row(dim);
                rows.push_back(row);
                residuals.push_back(linear(dim) - body_velocity(dim));
                variances.push_back(variance[dim]);
            }
        }
        const Vec3 predicted_angular = last_gyro_ - filter_.x.bg;
        for (int dim = 0; dim < 3; ++dim) {
            if (variance[3 + dim] > 0.0) {
                Vec15 row = Vec15::Zero();
                row(9 + dim) = -1.0;
                rows.push_back(row);
                residuals.push_back(angular(dim) - predicted_angular(dim));
                variances.push_back(variance[3 + dim]);
            }
        }
    }

    bool update(const std::vector<Vec15>& rows, const std::vector<double>& residuals,
                const std::vector<double>& variances) {
        if (rows.empty()) {
            return false;
        }
        const Eigen::Index count = static_cast<Eigen::Index>(rows.size());
        Eigen::VectorXd residual(count);
        Eigen::MatrixXd jacobian(count, 15);
        Eigen::VectorXd variance(count);
        for (Eigen::Index i = 0; i < count; ++i) {
            residual(i) = residuals[static_cast<std::size_t>(i)];
            jacobian.row(i) = rows[static_cast<std::size_t>(i)].transpose();
            variance(i) = variances[static_cast<std::size_t>(i)];
        }
        const bool accepted =
            filter_.update(residual, jacobian, variance, cfg_.mahalanobis_gate);
        if (!accepted) {
            ++gated_;
        }
        return accepted;
    }

    void anchor(std::size_t source) {
        Transform pose = Transform::Identity();
        pose.linear() = filter_.x.q.toRotationMatrix();
        pose.translation() = filter_.x.p;
        anchors_[source] = pose;
        anchored_[source] = true;
    }

    void snapshot(Event& event) {
        event.state = filter_.x;
        event.covariance = filter_.P;
        event.anchors = anchors_;
        event.anchored.assign(anchored_.begin(), anchored_.end());
        event.last_imu_ns = last_imu_ns_;
        event.last_gyro = last_gyro_;
    }

    void maybe_publish() {
        const Event& latest = events_.back();
        const std::int64_t period =
            static_cast<std::int64_t>(1.0e9 / std::max(cfg_.publish_rate, 1.0));
        if (latest.ts_ns - last_publish_ns_ < period) {
            return;
        }
        last_publish_ns_ = latest.ts_ns;
        const eskf::State& state = latest.state;
        const Eigen::Matrix3d body_from_world = state.q.conjugate().toRotationMatrix();

        nav_msgs::Odometry msg{};
        msg.header.stamp = to_stamp(latest.ts_ns);
        msg.header.frame_id = cfg_.odom_frame;
        msg.child_frame_id = cfg_.base_frame;
        msg.pose.pose.position.x = state.p.x();
        msg.pose.pose.position.y = state.p.y();
        msg.pose.pose.position.z = state.p.z();
        msg.pose.pose.orientation.x = state.q.x();
        msg.pose.pose.orientation.y = state.q.y();
        msg.pose.pose.orientation.z = state.q.z();
        msg.pose.pose.orientation.w = state.q.w();
        const Vec3 body_velocity = body_from_world * state.v;
        const Vec3 body_angular = latest.last_gyro - state.bg;
        msg.twist.twist.linear.x = body_velocity.x();
        msg.twist.twist.linear.y = body_velocity.y();
        msg.twist.twist.linear.z = body_velocity.z();
        msg.twist.twist.angular.x = body_angular.x();
        msg.twist.twist.angular.y = body_angular.y();
        msg.twist.twist.angular.z = body_angular.z();
        Eigen::Matrix<double, 6, 6> pose_covariance;
        pose_covariance.topLeftCorner<3, 3>() = latest.covariance.block<3, 3>(0, 0);
        pose_covariance.topRightCorner<3, 3>() = latest.covariance.block<3, 3>(0, 6);
        pose_covariance.bottomLeftCorner<3, 3>() = latest.covariance.block<3, 3>(6, 0);
        pose_covariance.bottomRightCorner<3, 3>() = latest.covariance.block<3, 3>(6, 6);
        Eigen::Matrix<double, 6, 6> twist_covariance =
            Eigen::Matrix<double, 6, 6>::Zero();
        twist_covariance.topLeftCorner<3, 3>() =
            body_from_world * latest.covariance.block<3, 3>(3, 3) *
            body_from_world.transpose();
        twist_covariance.bottomRightCorner<3, 3>() = latest.covariance.block<3, 3>(9, 9);
        for (int row = 0; row < 6; ++row) {
            for (int col = 0; col < 6; ++col) {
                msg.pose.covariance[row * 6 + col] = pose_covariance(row, col);
                msg.twist.covariance[row * 6 + col] = twist_covariance(row, col);
            }
        }
        odometry_.publish(msg);

        tf2_msgs::TFMessage tf_message{};
        geometry_msgs::TransformStamped stamped{};
        stamped.header.stamp = msg.header.stamp;
        stamped.header.frame_id = cfg_.odom_frame;
        stamped.child_frame_id = cfg_.base_frame;
        stamped.transform.translation.x = state.p.x();
        stamped.transform.translation.y = state.p.y();
        stamped.transform.translation.z = state.p.z();
        stamped.transform.rotation.x = state.q.x();
        stamped.transform.rotation.y = state.q.y();
        stamped.transform.rotation.z = state.q.z();
        stamped.transform.rotation.w = state.q.w();
        tf_message.transforms.push_back(stamped);
        tf_message.transforms_length = 1;
        tf_.publish(tf_message);
    }

    FusedOdometryConfig cfg_{};
    eskf::Filter filter_{};

    bool initialized_{false};
    Vec3 init_gyro_sum_ = Vec3::Zero();
    Vec3 init_accel_sum_ = Vec3::Zero();
    std::uint64_t init_samples_{0};

    std::deque<Event> events_;
    std::vector<Transform> anchors_;
    std::vector<bool> anchored_;
    std::int64_t last_imu_ns_{0};
    Vec3 last_gyro_ = Vec3::Zero();
    /// Per-source last raw pose; deltas are a property of the message stream, so they
    /// are computed once at arrival and survive replay untouched.
    std::unordered_map<std::size_t, std::optional<Transform>> previous_source_pose_;

    std::int64_t last_publish_ns_{0};
    std::uint64_t imu_samples_{0};
    std::uint64_t measurements_{0};
    std::uint64_t gated_{0};
    std::uint64_t replayed_{0};
    std::uint64_t too_late_{0};

    Output<nav_msgs::Odometry> odometry_;
    Output<tf2_msgs::TFMessage> tf_;
};

int main() {
    dimos::native::run_with_transport<FusedOdometry>();
    return 0;
}
