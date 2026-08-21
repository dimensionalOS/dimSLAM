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

use dimos_module::nalgebra::{DVector, Dyn, Matrix3, OMatrix, SMatrix, UnitQuaternion, Vector3, U15};

pub type Mat15 = SMatrix<f64, 15, 15>;
pub type Vec15 = SMatrix<f64, 15, 1>;
pub type Jacobian = OMatrix<f64, Dyn, U15>;

pub fn skew(v: &Vector3<f64>) -> Matrix3<f64> {
    v.cross_matrix()
}

#[derive(Clone, Copy, Debug, Default)]
pub struct Noise {
    pub gyro_noise_density: f64,   // rad/s/sqrt(Hz)
    pub gyro_random_walk: f64,     // rad/s^2/sqrt(Hz)
    pub accel_noise_density: f64,  // m/s^2/sqrt(Hz)
    pub accel_random_walk: f64,    // m/s^3/sqrt(Hz)
}

#[derive(Clone, Debug)]
pub struct State {
    pub p: Vector3<f64>,           // world position of the body
    pub v: Vector3<f64>,           // world velocity
    pub q: UnitQuaternion<f64>,    // world_from_body
    pub bg: Vector3<f64>,          // gyro bias
    pub ba: Vector3<f64>,          // accel bias
}

impl Default for State {
    fn default() -> Self {
        Self {
            p: Vector3::zeros(),
            v: Vector3::zeros(),
            q: UnitQuaternion::identity(),
            bg: Vector3::zeros(),
            ba: Vector3::zeros(),
        }
    }
}

#[derive(Clone, Debug)]
pub struct Filter {
    pub noise: Noise,
    pub gravity: f64,
    pub x: State,
    pub p_cov: Mat15,
}

impl Default for Filter {
    fn default() -> Self {
        Self {
            noise: Noise::default(),
            gravity: 9.80665,
            x: State::default(),
            p_cov: Mat15::identity(),
        }
    }
}

impl Filter {
    #[allow(clippy::too_many_arguments)]
    pub fn init(
        &mut self,
        world_from_body: UnitQuaternion<f64>,
        gyro_bias: Vector3<f64>,
        accel_bias: Vector3<f64>,
        position_std: f64,
        velocity_std: f64,
        rotation_std: f64,
        bias_std: f64,
    ) {
        self.x = State::default();
        self.x.q = world_from_body;
        self.x.bg = gyro_bias;
        self.x.ba = accel_bias;
        self.p_cov = Mat15::zeros();
        for (block, std) in [
            (0, position_std),
            (3, velocity_std),
            (6, rotation_std),
            (9, bias_std),
            (12, bias_std),
        ] {
            self.p_cov
                .fixed_view_mut::<3, 3>(block, block)
                .copy_from(&(Matrix3::identity() * std * std));
        }
    }

    pub fn propagate(&mut self, dt: f64, gyro: &Vector3<f64>, accel: &Vector3<f64>) {
        let rotation = self.x.q.to_rotation_matrix();
        let unbiased_gyro = gyro - self.x.bg;
        let unbiased_accel = accel - self.x.ba;
        let gravity_vector = Vector3::new(0.0, 0.0, -self.gravity);
        let world_accel = rotation * unbiased_accel + gravity_vector;

        let mut f = Mat15::identity();
        f.fixed_view_mut::<3, 3>(0, 3)
            .copy_from(&(Matrix3::identity() * dt));
        f.fixed_view_mut::<3, 3>(3, 6)
            .copy_from(&(-(rotation.matrix() * skew(&unbiased_accel)) * dt));
        f.fixed_view_mut::<3, 3>(3, 12)
            .copy_from(&(-rotation.matrix() * dt));
        f.fixed_view_mut::<3, 3>(6, 6)
            .copy_from(&(Matrix3::identity() - skew(&unbiased_gyro) * dt));
        f.fixed_view_mut::<3, 3>(6, 9)
            .copy_from(&(-Matrix3::identity() * dt));

        self.x.p += self.x.v * dt + 0.5 * world_accel * dt * dt;
        self.x.v += world_accel * dt;
        self.x.q *= UnitQuaternion::from_scaled_axis(unbiased_gyro * dt);

        let mut q_noise = Mat15::zeros();
        for (block, density) in [
            (3, self.noise.accel_noise_density),
            (6, self.noise.gyro_noise_density),
            (9, self.noise.gyro_random_walk),
            (12, self.noise.accel_random_walk),
        ] {
            q_noise
                .fixed_view_mut::<3, 3>(block, block)
                .copy_from(&(Matrix3::identity() * density * density * dt));
        }
        self.p_cov = f * self.p_cov * f.transpose() + q_noise;
    }

    /// One stacked update with a diagonal noise. `gate` is a Mahalanobis
    /// threshold in standard deviations per degree of freedom; 0 disables the
    /// gate. Returns false when the measurement was rejected.
    // The negated comparison is deliberate: a NaN distance must reject.
    #[allow(clippy::neg_cmp_op_on_partial_ord)]
    pub fn update(
        &mut self,
        residual: &DVector<f64>,
        jacobian: &Jacobian,
        variance: &DVector<f64>,
        gate: f64,
    ) -> bool {
        let noise = OMatrix::<f64, Dyn, Dyn>::from_diagonal(variance);
        let innovation = jacobian * self.p_cov * jacobian.transpose() + &noise;
        let Some(innovation_inverse) = innovation.try_inverse() else {
            return false;
        };
        if gate > 0.0 {
            let mahalanobis_sq = (residual.transpose() * &innovation_inverse * residual)[(0, 0)];
            if !(mahalanobis_sq < gate * gate * residual.len() as f64) {
                return false;
            }
        }
        let kalman_gain = self.p_cov * jacobian.transpose() * &innovation_inverse;
        let dx: Vec15 = &kalman_gain * residual;
        let identity_minus_kh: Mat15 = Mat15::identity() - &kalman_gain * jacobian;
        self.p_cov = identity_minus_kh * self.p_cov * identity_minus_kh.transpose()
            + &kalman_gain * noise * kalman_gain.transpose();
        self.inject(&dx);
        true
    }

    fn inject(&mut self, dx: &Vec15) {
        self.x.p += dx.fixed_rows::<3>(0);
        self.x.v += dx.fixed_rows::<3>(3);
        self.x.q *= UnitQuaternion::from_scaled_axis(dx.fixed_rows::<3>(6).into_owned());
        self.x.bg += dx.fixed_rows::<3>(9);
        self.x.ba += dx.fixed_rows::<3>(12);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const GRAVITY: f64 = 9.80665;

    fn level_filter() -> Filter {
        let mut filter = Filter {
            noise: Noise {
                gyro_noise_density: 0.01,
                gyro_random_walk: 0.001,
                accel_noise_density: 0.1,
                accel_random_walk: 0.01,
            },
            ..Filter::default()
        };
        filter.init(
            UnitQuaternion::identity(),
            Vector3::zeros(),
            Vector3::zeros(),
            0.1,
            0.1,
            0.05,
            0.01,
        );
        filter
    }

    /// Gravity-compensating accel for a level, stationary body.
    fn static_accel() -> Vector3<f64> {
        Vector3::new(0.0, 0.0, GRAVITY)
    }

    fn position_jacobian(rows: usize) -> Jacobian {
        let mut jacobian = Jacobian::zeros(rows);
        jacobian.fixed_view_mut::<3, 3>(0, 0).fill_with_identity();
        jacobian
    }

    #[test]
    fn skew_is_antisymmetric_and_reproduces_the_cross_product() {
        let a = Vector3::new(1.0, 2.0, 3.0);
        let b = Vector3::new(-0.5, 4.0, 0.25);
        let a_skew = skew(&a);
        assert!((a_skew + a_skew.transpose()).norm() < 1e-12);
        assert!((a_skew * b - a.cross(&b)).norm() < 1e-12);
    }

    #[test]
    fn exp_and_log_of_so3_round_trip() {
        let rotation = Vector3::new(0.3, -0.2, 0.9);
        let q = UnitQuaternion::from_scaled_axis(rotation);
        assert!((q.scaled_axis() - rotation).norm() < 1e-9);
    }

    #[test]
    fn exp_of_a_tiny_rotation_stays_finite_and_normalized() {
        let q = UnitQuaternion::from_scaled_axis(Vector3::new(1e-14, 0.0, 0.0));
        let norm: f64 = q.norm();
        assert!((norm - 1.0).abs() < 1e-12);
    }

    #[test]
    fn stationary_propagation_holds_the_state() {
        let mut filter = level_filter();
        for _ in 0..100 {
            filter.propagate(0.01, &Vector3::zeros(), &static_accel());
        }
        assert!(filter.x.p.norm() < 1e-9);
        assert!(filter.x.v.norm() < 1e-9);
    }

    #[test]
    fn propagation_integrates_constant_velocity() {
        let mut filter = level_filter();
        filter.x.v = Vector3::new(1.0, 0.0, 0.0);
        for _ in 0..100 {
            filter.propagate(0.01, &Vector3::zeros(), &static_accel());
        }
        assert!((filter.x.p.x - 1.0).abs() < 1e-9);
    }

    #[test]
    fn propagation_grows_uncertainty() {
        let mut filter = level_filter();
        let before = filter.p_cov.trace();
        for _ in 0..100 {
            filter.propagate(0.01, &Vector3::zeros(), &static_accel());
        }
        assert!(filter.p_cov.trace() > before);
    }

    #[test]
    fn a_position_update_pulls_the_estimate() {
        let mut filter = level_filter();
        filter.propagate(0.01, &Vector3::zeros(), &static_accel());
        let residual = DVector::from_vec(vec![1.0, 0.0, 0.0]);
        let variance = DVector::from_element(3, 1e-4);
        assert!(filter.update(&residual, &position_jacobian(3), &variance, 0.0));
        assert!(filter.x.p.x > 0.9);
        assert!(filter.p_cov.trace() < 15.0);
    }

    #[test]
    fn position_updates_correct_velocity_through_the_cross_covariance() {
        let mut filter = level_filter();
        filter.x.v = Vector3::new(1.0, 0.0, 0.0); // true velocity is zero; the filter is wrong
        for _ in 0..50 {
            filter.propagate(0.01, &Vector3::zeros(), &static_accel());
            // The truth stays at the origin, so the residual is minus the estimate.
            let residual = DVector::from_vec(vec![-filter.x.p.x, -filter.x.p.y, -filter.x.p.z]);
            let variance = DVector::from_element(3, 1e-6);
            filter.update(&residual, &position_jacobian(3), &variance, 0.0);
        }
        assert!(filter.x.v.x.abs() < 0.1);
    }

    #[test]
    fn the_mahalanobis_gate_rejects_an_outlier_and_passes_an_inlier() {
        let mut filter = level_filter();
        filter.propagate(0.01, &Vector3::zeros(), &static_accel());
        let variance = DVector::from_element(3, 1e-4);
        let outlier = DVector::from_vec(vec![100.0, 0.0, 0.0]);
        assert!(!filter.update(&outlier, &position_jacobian(3), &variance, 3.0));
        let inlier = DVector::from_vec(vec![0.01, 0.0, 0.0]);
        assert!(filter.update(&inlier, &position_jacobian(3), &variance, 3.0));
    }

    #[test]
    fn a_gate_of_zero_disables_rejection() {
        let mut filter = level_filter();
        let variance = DVector::from_element(3, 1e-4);
        let outlier = DVector::from_vec(vec![100.0, 0.0, 0.0]);
        assert!(filter.update(&outlier, &position_jacobian(3), &variance, 0.0));
    }

    #[test]
    fn a_gyro_bias_propagates_into_a_tilt_the_accel_residual_can_expose() {
        let mut filter = level_filter();
        let true_bias = Vector3::new(0.02, 0.0, 0.0);
        // The gyro reads only its bias while the body is still: the estimate tilts.
        for _ in 0..100 {
            filter.propagate(0.01, &true_bias, &static_accel());
        }
        let tilt = filter.x.q.scaled_axis();
        assert!((tilt.x - 0.02).abs() < 1e-3);
    }

    #[test]
    fn an_angular_rate_update_observes_the_gyro_bias() {
        let mut filter = level_filter();
        let gyro_reading = Vector3::new(0.05, 0.0, 0.0); // still body, biased gyro
        for _ in 0..50 {
            filter.propagate(0.01, &gyro_reading, &static_accel());
            // A perfect rate sensor says the body is not rotating: residual is
            // 0 - (gyro - bg), jacobian -I on the bias block.
            let predicted = gyro_reading - filter.x.bg;
            let residual = DVector::from_vec(vec![-predicted.x, -predicted.y, -predicted.z]);
            let mut jacobian = Jacobian::zeros(3);
            jacobian
                .fixed_view_mut::<3, 3>(0, 9)
                .copy_from(&(-Matrix3::identity()));
            let variance = DVector::from_element(3, 1e-6);
            filter.update(&residual, &jacobian, &variance, 0.0);
        }
        assert!((filter.x.bg.x - 0.05).abs() < 1e-3);
    }

    #[test]
    fn init_seeds_the_covariance_from_the_stds() {
        let mut filter = Filter::default();
        filter.init(
            UnitQuaternion::identity(),
            Vector3::new(0.1, 0.0, 0.0),
            Vector3::zeros(),
            2.0,
            3.0,
            0.5,
            0.01,
        );
        assert!((filter.p_cov[(0, 0)] - 4.0).abs() < 1e-12);
        assert!((filter.p_cov[(3, 3)] - 9.0).abs() < 1e-12);
        assert!((filter.p_cov[(6, 6)] - 0.25).abs() < 1e-12);
        assert!((filter.p_cov[(9, 9)] - 1e-4).abs() < 1e-12);
        assert!((filter.x.bg.x - 0.1).abs() < 1e-12);
    }
}
