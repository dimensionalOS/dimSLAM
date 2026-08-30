// Copyright 2026 Dimensional Inc.
// SPDX-License-Identifier: Apache-2.0
//
// Error-state order: position(0:3) velocity(3:6) theta(6:9), then a gyro_bias/accel_bias
// pair per IMU at (9+6i : 15+6i). theta is a right (body-frame) perturbation:
// q_true = q_est * Exp(theta).

use nalgebra::{DMatrix, DVector, Matrix3, UnitQuaternion, Vector3};

pub type Cov = DMatrix<f64>;
pub type Jacobian = DMatrix<f64>;

pub fn skew(v: &Vector3<f64>) -> Matrix3<f64> {
    v.cross_matrix()
}

pub fn gyro_bias_col(imu: usize) -> usize {
    9 + 6 * imu
}

pub fn accel_bias_col(imu: usize) -> usize {
    12 + 6 * imu
}

#[derive(Clone, Copy, Debug, Default)]
pub struct Noise {
    pub gyro_noise_density: f64,  // rad/s/sqrt(Hz)
    pub gyro_random_walk: f64,    // rad/s^2/sqrt(Hz)
    pub accel_noise_density: f64, // m/s^2/sqrt(Hz)
    pub accel_random_walk: f64,   // m/s^3/sqrt(Hz)
}

#[derive(Clone, Debug, Default)]
pub struct ImuBias {
    pub gyro: Vector3<f64>,
    pub accel: Vector3<f64>,
}

#[derive(Clone, Debug)]
pub struct State {
    pub position: Vector3<f64>, // world frame
    pub velocity: Vector3<f64>, // world frame
    pub q: UnitQuaternion<f64>, // world_from_body
    pub biases: Vec<ImuBias>,
}

impl Default for State {
    fn default() -> Self {
        Self {
            position: Vector3::zeros(),
            velocity: Vector3::zeros(),
            q: UnitQuaternion::identity(),
            biases: Vec::new(),
        }
    }
}

#[derive(Clone, Debug)]
pub struct Filter {
    /// One entry per IMU, parallel to `x.biases`.
    pub noise: Vec<Noise>,
    pub gravity: f64,
    pub x: State,
    pub p_cov: Cov,
}

impl Default for Filter {
    fn default() -> Self {
        Self {
            noise: Vec::new(),
            gravity: 9.80665,
            x: State::default(),
            p_cov: Cov::identity(9, 9),
        }
    }
}

impl Filter {
    pub fn dim(&self) -> usize {
        9 + 6 * self.x.biases.len()
    }

    pub fn init(
        &mut self,
        world_from_body: UnitQuaternion<f64>,
        biases: Vec<ImuBias>,
        position_std: f64,
        velocity_std: f64,
        rotation_std: f64,
        bias_std: f64,
    ) {
        self.x = State {
            q: world_from_body,
            biases,
            ..State::default()
        };
        let dim = self.dim();
        self.p_cov = Cov::zeros(dim, dim);
        let mut blocks = vec![(0, position_std), (3, velocity_std), (6, rotation_std)];
        for imu in 0..self.x.biases.len() {
            blocks.push((gyro_bias_col(imu), bias_std));
            blocks.push((accel_bias_col(imu), bias_std));
        }
        for (block, std) in blocks {
            self.p_cov
                .view_mut((block, block), (3, 3))
                .copy_from(&(Matrix3::identity() * std * std));
        }
    }

    /// One step of dead reckoning from IMU `imu`'s sample, using that IMU's bias and noise.
    /// With interleaved IMUs each sample integrates the slice of time since the previous
    /// sample of any IMU.
    pub fn propagate(&mut self, dt: f64, imu: usize, gyro: &Vector3<f64>, accel: &Vector3<f64>) {
        let dim = self.dim();
        let rotation = self.x.q.to_rotation_matrix();
        let unbiased_gyro = gyro - self.x.biases[imu].gyro;
        let unbiased_accel = accel - self.x.biases[imu].accel;
        let gravity_vector = Vector3::new(0.0, 0.0, -self.gravity);
        let world_accel = rotation * unbiased_accel + gravity_vector;

        let mut f = Cov::identity(dim, dim);
        f.view_mut((0, 3), (3, 3))
            .copy_from(&(Matrix3::identity() * dt));
        f.view_mut((3, 6), (3, 3))
            .copy_from(&(-(rotation.matrix() * skew(&unbiased_accel)) * dt));
        f.view_mut((3, accel_bias_col(imu)), (3, 3))
            .copy_from(&(-rotation.matrix() * dt));
        f.view_mut((6, 6), (3, 3))
            .copy_from(&(Matrix3::identity() - skew(&unbiased_gyro) * dt));
        f.view_mut((6, gyro_bias_col(imu)), (3, 3))
            .copy_from(&(-Matrix3::identity() * dt));

        self.x.position += self.x.velocity * dt + 0.5 * world_accel * dt * dt;
        self.x.velocity += world_accel * dt;
        self.x.q *= UnitQuaternion::from_scaled_axis(unbiased_gyro * dt);

        let mut q_noise = Cov::zeros(dim, dim);
        let mut blocks = vec![
            (3, self.noise[imu].accel_noise_density),
            (6, self.noise[imu].gyro_noise_density),
        ];
        // Every IMU's bias random-walks through the interval, not just the one sampling.
        for (other, noise) in self.noise.iter().enumerate() {
            blocks.push((gyro_bias_col(other), noise.gyro_random_walk));
            blocks.push((accel_bias_col(other), noise.accel_random_walk));
        }
        for (block, density) in blocks {
            q_noise
                .view_mut((block, block), (3, 3))
                .copy_from(&(Matrix3::identity() * density * density * dt));
        }
        self.p_cov = &f * &self.p_cov * f.transpose() + q_noise;
    }

    /// `gate` is a Mahalanobis threshold in standard deviations per degree of freedom; 0 disables.
    // The negated comparison is deliberate: a NaN distance must reject.
    #[allow(clippy::neg_cmp_op_on_partial_ord)]
    pub fn update(
        &mut self,
        residual: &DVector<f64>,
        jacobian: &Jacobian,
        variance: &DVector<f64>,
        gate: f64,
    ) -> bool {
        let noise = Cov::from_diagonal(variance);
        let innovation = jacobian * &self.p_cov * jacobian.transpose() + &noise;
        let Some(innovation_inverse) = innovation.try_inverse() else {
            return false;
        };
        if gate > 0.0 {
            let mahalanobis_sq = (residual.transpose() * &innovation_inverse * residual)[(0, 0)];
            if !(mahalanobis_sq < gate * gate * residual.len() as f64) {
                return false;
            }
        }
        let kalman_gain = &self.p_cov * jacobian.transpose() * &innovation_inverse;
        let dx: DVector<f64> = &kalman_gain * residual;
        let identity_minus_kh = Cov::identity(self.dim(), self.dim()) - &kalman_gain * jacobian;
        self.p_cov = &identity_minus_kh * &self.p_cov * identity_minus_kh.transpose()
            + &kalman_gain * noise * kalman_gain.transpose();
        self.inject(&dx);
        true
    }

    fn inject(&mut self, dx: &DVector<f64>) {
        self.x.position += dx.fixed_rows::<3>(0);
        self.x.velocity += dx.fixed_rows::<3>(3);
        self.x.q *= UnitQuaternion::from_scaled_axis(dx.fixed_rows::<3>(6).into_owned());
        for (imu, bias) in self.x.biases.iter_mut().enumerate() {
            bias.gyro += dx.fixed_rows::<3>(gyro_bias_col(imu));
            bias.accel += dx.fixed_rows::<3>(accel_bias_col(imu));
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const GRAVITY: f64 = 9.80665;

    fn level_filter() -> Filter {
        let mut filter = Filter {
            noise: vec![Noise {
                gyro_noise_density: 0.01,
                gyro_random_walk: 0.001,
                accel_noise_density: 0.1,
                accel_random_walk: 0.01,
            }],
            ..Filter::default()
        };
        filter.init(
            UnitQuaternion::identity(),
            vec![ImuBias::default()],
            0.1,
            0.1,
            0.05,
            0.01,
        );
        filter
    }

    fn static_accel() -> Vector3<f64> {
        Vector3::new(0.0, 0.0, GRAVITY)
    }

    fn position_jacobian(rows: usize, dim: usize) -> Jacobian {
        let mut jacobian = Jacobian::zeros(rows, dim);
        jacobian.view_mut((0, 0), (3, 3)).fill_with_identity();
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
            filter.propagate(0.01, 0, &Vector3::zeros(), &static_accel());
        }
        assert!(filter.x.position.norm() < 1e-9);
        assert!(filter.x.velocity.norm() < 1e-9);
    }

    #[test]
    fn propagation_integrates_constant_velocity() {
        let mut filter = level_filter();
        filter.x.velocity = Vector3::new(1.0, 0.0, 0.0);
        for _ in 0..100 {
            filter.propagate(0.01, 0, &Vector3::zeros(), &static_accel());
        }
        assert!((filter.x.position.x - 1.0).abs() < 1e-9);
    }

    #[test]
    fn propagation_grows_uncertainty() {
        let mut filter = level_filter();
        let before = filter.p_cov.trace();
        for _ in 0..100 {
            filter.propagate(0.01, 0, &Vector3::zeros(), &static_accel());
        }
        assert!(filter.p_cov.trace() > before);
    }

    #[test]
    fn a_position_update_pulls_the_estimate() {
        let mut filter = level_filter();
        filter.propagate(0.01, 0, &Vector3::zeros(), &static_accel());
        let residual = DVector::from_vec(vec![1.0, 0.0, 0.0]);
        let variance = DVector::from_element(3, 1e-4);
        assert!(filter.update(&residual, &position_jacobian(3, filter.dim()), &variance, 0.0));
        assert!(filter.x.position.x > 0.9);
        assert!(filter.p_cov.trace() < 15.0);
    }

    #[test]
    fn position_updates_correct_velocity_through_the_cross_covariance() {
        let mut filter = level_filter();
        filter.x.velocity = Vector3::new(1.0, 0.0, 0.0); // true velocity is zero; the filter is wrong
        for _ in 0..50 {
            filter.propagate(0.01, 0, &Vector3::zeros(), &static_accel());
            // The truth stays at the origin, so the residual is minus the estimate.
            let residual = DVector::from_vec(vec![
                -filter.x.position.x,
                -filter.x.position.y,
                -filter.x.position.z,
            ]);
            let variance = DVector::from_element(3, 1e-6);
            filter.update(&residual, &position_jacobian(3, filter.dim()), &variance, 0.0);
        }
        assert!(filter.x.velocity.x.abs() < 0.1);
    }

    #[test]
    fn the_mahalanobis_gate_rejects_an_outlier_and_passes_an_inlier() {
        let mut filter = level_filter();
        filter.propagate(0.01, 0, &Vector3::zeros(), &static_accel());
        let variance = DVector::from_element(3, 1e-4);
        let outlier = DVector::from_vec(vec![100.0, 0.0, 0.0]);
        assert!(!filter.update(&outlier, &position_jacobian(3, filter.dim()), &variance, 3.0));
        let inlier = DVector::from_vec(vec![0.01, 0.0, 0.0]);
        assert!(filter.update(&inlier, &position_jacobian(3, filter.dim()), &variance, 3.0));
    }

    #[test]
    fn a_gate_of_zero_disables_rejection() {
        let mut filter = level_filter();
        let variance = DVector::from_element(3, 1e-4);
        let outlier = DVector::from_vec(vec![100.0, 0.0, 0.0]);
        assert!(filter.update(&outlier, &position_jacobian(3, filter.dim()), &variance, 0.0));
    }

    #[test]
    fn a_gyro_bias_propagates_into_a_tilt_the_accel_residual_can_expose() {
        let mut filter = level_filter();
        let true_bias = Vector3::new(0.02, 0.0, 0.0);
        // The gyro reads only its bias while the body is still: the estimate tilts.
        for _ in 0..100 {
            filter.propagate(0.01, 0, &true_bias, &static_accel());
        }
        let tilt = filter.x.q.scaled_axis();
        assert!((tilt.x - 0.02).abs() < 1e-3);
    }

    #[test]
    fn an_angular_rate_update_observes_the_gyro_bias() {
        let mut filter = level_filter();
        let gyro_reading = Vector3::new(0.05, 0.0, 0.0); // still body, biased gyro
        for _ in 0..50 {
            filter.propagate(0.01, 0, &gyro_reading, &static_accel());
            // A perfect rate sensor says the body is not rotating.
            let predicted = gyro_reading - filter.x.biases[0].gyro;
            let residual = DVector::from_vec(vec![-predicted.x, -predicted.y, -predicted.z]);
            let mut jacobian = Jacobian::zeros(3, filter.dim());
            jacobian
                .view_mut((0, gyro_bias_col(0)), (3, 3))
                .copy_from(&(-Matrix3::identity()));
            let variance = DVector::from_element(3, 1e-6);
            filter.update(&residual, &jacobian, &variance, 0.0);
        }
        assert!((filter.x.biases[0].gyro.x - 0.05).abs() < 1e-3);
    }

    #[test]
    fn init_seeds_the_covariance_from_the_stds() {
        let mut filter = Filter {
            noise: vec![Noise::default()],
            ..Filter::default()
        };
        filter.init(
            UnitQuaternion::identity(),
            vec![ImuBias {
                gyro: Vector3::new(0.1, 0.0, 0.0),
                accel: Vector3::zeros(),
            }],
            2.0,
            3.0,
            0.5,
            0.01,
        );
        assert!((filter.p_cov[(0, 0)] - 4.0).abs() < 1e-12);
        assert!((filter.p_cov[(3, 3)] - 9.0).abs() < 1e-12);
        assert!((filter.p_cov[(6, 6)] - 0.25).abs() < 1e-12);
        assert!((filter.p_cov[(9, 9)] - 1e-4).abs() < 1e-12);
        assert!((filter.x.biases[0].gyro.x - 0.1).abs() < 1e-12);
    }

    #[test]
    fn two_imus_keep_separate_bias_states() {
        let noise = Noise {
            gyro_noise_density: 0.01,
            gyro_random_walk: 0.001,
            accel_noise_density: 0.1,
            accel_random_walk: 0.01,
        };
        let mut filter = Filter {
            noise: vec![noise, noise],
            ..Filter::default()
        };
        filter.init(
            UnitQuaternion::identity(),
            vec![ImuBias::default(), ImuBias::default()],
            0.1,
            0.1,
            0.05,
            0.01,
        );
        assert_eq!(filter.dim(), 21);
        // Only the second IMU's gyro is biased; a rate update on its block finds it there.
        let gyro_reading = Vector3::new(0.05, 0.0, 0.0);
        for step in 0..100 {
            if step % 2 == 0 {
                filter.propagate(0.005, 0, &Vector3::zeros(), &static_accel());
            } else {
                filter.propagate(0.005, 1, &gyro_reading, &static_accel());
                let predicted = gyro_reading - filter.x.biases[1].gyro;
                let residual = DVector::from_vec(vec![-predicted.x, -predicted.y, -predicted.z]);
                let mut jacobian = Jacobian::zeros(3, filter.dim());
                jacobian
                    .view_mut((0, gyro_bias_col(1)), (3, 3))
                    .copy_from(&(-Matrix3::identity()));
                let variance = DVector::from_element(3, 1e-6);
                filter.update(&residual, &jacobian, &variance, 0.0);
            }
        }
        assert!((filter.x.biases[1].gyro.x - 0.05).abs() < 2e-3);
        assert!(filter.x.biases[0].gyro.x.abs() < 2e-2);
    }
}
