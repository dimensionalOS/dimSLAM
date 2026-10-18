// Copyright 2026 Dimensional Inc.
// SPDX-License-Identifier: Apache-2.0
//
// Error-state order: position(0:3) velocity(3:6) theta(6:9), then angular velocity(9:12) when
// there is no IMU, else a gyro_bias/accel_bias pair per IMU, then a position/theta pair per
// cloned pose. theta is a right (body-frame) perturbation: q_true = q_est * Exp(theta).

use nalgebra::{DMatrix, DVector, Matrix3, UnitQuaternion, Vector3};

use super::{InitialStds, ProcessNoise};

pub type Cov = DMatrix<f64>;
pub type Jacobian = DMatrix<f64>;

pub fn skew(v: &Vector3<f64>) -> Matrix3<f64> {
    v.cross_matrix()
}

/// Rotation about z alone: what a planar body's orientation is allowed to be.
pub fn yaw_only(q: &UnitQuaternion<f64>) -> UnitQuaternion<f64> {
    let (_, _, yaw) = q.euler_angles();
    UnitQuaternion::from_axis_angle(&Vector3::z_axis(), yaw)
}

/// Where each block lives in the error state; fixed at init.
#[derive(Clone, Copy, Debug, Default)]
pub struct Layout {
    pub imus: usize,
    pub clones: usize,
}

impl Layout {
    /// Without an IMU the angular rate is a state; with one it is the gyro reading.
    pub fn angular_velocity_col(&self) -> Option<usize> {
        (self.imus == 0).then_some(9)
    }

    pub fn gyro_bias_col(&self, imu: usize) -> usize {
        9 + 6 * imu
    }

    pub fn accel_bias_col(&self, imu: usize) -> usize {
        12 + 6 * imu
    }

    fn clones_start(&self) -> usize {
        if self.imus == 0 {
            12
        } else {
            9 + 6 * self.imus
        }
    }

    pub fn clone_position_col(&self, slot: usize) -> usize {
        self.clones_start() + 6 * slot
    }

    pub fn clone_rotation_col(&self, slot: usize) -> usize {
        self.clone_position_col(slot) + 3
    }

    pub fn dim(&self) -> usize {
        self.clones_start() + 6 * self.clones
    }

    /// The dimensions a planar body cannot move in: z, roll, pitch and their rates.
    fn pinned_cols(&self) -> Vec<usize> {
        let mut cols = vec![2, 5, 6, 7];
        if let Some(col) = self.angular_velocity_col() {
            cols.extend([col, col + 1]);
        }
        for slot in 0..self.clones {
            let position = self.clone_position_col(slot);
            let rotation = self.clone_rotation_col(slot);
            cols.extend([position + 2, rotation, rotation + 1]);
        }
        cols
    }
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

/// The latest IMU sample, base-frame, held as the process input until the next one.
#[derive(Clone, Debug)]
pub struct ImuInput {
    pub imu: usize,
    pub gyro: Vector3<f64>,
    pub accel: Vector3<f64>,
}

/// A copy of the pose at some past time, so a relative measurement can span to the present.
#[derive(Clone, Debug)]
pub struct PoseClone {
    pub position: Vector3<f64>,
    pub q: UnitQuaternion<f64>,
}

#[derive(Clone, Debug)]
pub struct State {
    /// The instant the rest of the state describes.
    pub time_ns: i64,
    pub position: Vector3<f64>, // world frame
    pub velocity: Vector3<f64>, // world frame
    pub q: UnitQuaternion<f64>, // world_from_body
    /// Body frame; a state only without an IMU, otherwise unused.
    pub angular_velocity: Vector3<f64>,
    pub biases: Vec<ImuBias>,
    pub input: Option<ImuInput>,
    pub clones: Vec<Option<PoseClone>>,
    /// A twist source said the body is still: the constant-velocity model holds it without noise.
    pub still: bool,
}

impl Default for State {
    fn default() -> Self {
        Self {
            time_ns: 0,
            position: Vector3::zeros(),
            velocity: Vector3::zeros(),
            q: UnitQuaternion::identity(),
            angular_velocity: Vector3::zeros(),
            biases: Vec::new(),
            input: None,
            clones: Vec::new(),
            still: false,
        }
    }
}

/// Past this an IMU gap is a dropout, and integrating one held sample across it would invent motion.
const MAX_INERTIAL_STEP_SECONDS: f64 = 1.0;

#[derive(Clone, Debug, Default)]
pub struct Filter {
    pub layout: Layout,
    /// One entry per IMU, parallel to `x.biases`.
    pub noise: Vec<Noise>,
    pub process_noise: ProcessNoise,
    pub gravity: f64,
    pub planar: bool,
    pub x: State,
    pub p_cov: Cov,
}

impl Filter {
    pub fn dim(&self) -> usize {
        self.layout.dim()
    }

    pub fn init(
        &mut self,
        time_ns: i64,
        world_from_body: UnitQuaternion<f64>,
        biases: Vec<ImuBias>,
        input: Option<ImuInput>,
        stds: &InitialStds,
    ) {
        self.layout.imus = biases.len();
        self.x = State {
            time_ns,
            q: world_from_body,
            biases,
            input,
            clones: vec![None; self.layout.clones],
            ..State::default()
        };
        let dim = self.dim();
        self.p_cov = Cov::zeros(dim, dim);
        let mut blocks = vec![(0, stds.position), (3, stds.velocity), (6, stds.rotation)];
        if let Some(col) = self.layout.angular_velocity_col() {
            blocks.push((col, stds.angular_velocity));
        }
        for imu in 0..self.layout.imus {
            blocks.push((self.layout.gyro_bias_col(imu), stds.bias));
            blocks.push((self.layout.accel_bias_col(imu), stds.bias));
        }
        for (block, std) in blocks {
            self.p_cov
                .view_mut((block, block), (3, 3))
                .copy_from(&(Matrix3::identity() * std * std));
        }
        self.constrain();
    }

    /// Body-frame angular rate and the error-state column that corrects it, with its sign:
    /// the rate state itself, or minus the gyro bias of the IMU whose sample is held.
    pub fn angular_velocity(&self) -> (Vector3<f64>, Option<(usize, f64)>) {
        if let Some(col) = self.layout.angular_velocity_col() {
            return (self.x.angular_velocity, Some((col, 1.0)));
        }
        match &self.x.input {
            Some(input) => (
                input.gyro - self.x.biases[input.imu].gyro,
                Some((self.layout.gyro_bias_col(input.imu), -1.0)),
            ),
            None => (Vector3::zeros(), None),
        }
    }

    /// Moves the state to `time_ns` under the process model: the held IMU sample when there
    /// is an IMU, constant velocity otherwise. Returns false over an IMU gap, left unbridged.
    pub fn predict_to(&mut self, time_ns: i64) -> bool {
        let dt = (time_ns - self.x.time_ns) as f64 / 1.0e9;
        if dt <= 0.0 {
            return true;
        }
        self.x.time_ns = time_ns;
        let bridged = if self.layout.imus == 0 {
            self.predict_constant_velocity(dt);
            true
        } else if let Some(input) = self
            .x
            .input
            .clone()
            .filter(|_| dt <= MAX_INERTIAL_STEP_SECONDS)
        {
            self.predict_inertial(dt, input.imu, &input.gyro, &input.accel);
            true
        } else {
            false
        };
        self.constrain();
        bridged
    }

    fn predict_constant_velocity(&mut self, dt: f64) {
        if self.x.still {
            return;
        }
        let dim = self.dim();
        let omega_col = self.layout.angular_velocity_col().expect("no imu");
        let rotation_step = UnitQuaternion::from_scaled_axis(self.x.angular_velocity * dt);

        let mut f = Cov::identity(dim, dim);
        f.view_mut((0, 3), (3, 3))
            .copy_from(&(Matrix3::identity() * dt));
        f.view_mut((6, 6), (3, 3))
            .copy_from(rotation_step.inverse().to_rotation_matrix().matrix());
        f.view_mut((6, omega_col), (3, 3))
            .copy_from(&(Matrix3::identity() * dt));

        self.x.position += self.x.velocity * dt;
        self.x.q *= rotation_step;

        // Integrated white acceleration: the position-velocity pair is correlated, not independent.
        let mut q_noise = Cov::zeros(dim, dim);
        for (value_col, rate_col, density) in [
            (0, 3, self.process_noise.linear_acceleration),
            (6, omega_col, self.process_noise.angular_acceleration),
        ] {
            let spectral = density * density;
            let identity = Matrix3::identity();
            q_noise
                .view_mut((value_col, value_col), (3, 3))
                .copy_from(&(identity * spectral * dt.powi(3) / 3.0));
            q_noise
                .view_mut((value_col, rate_col), (3, 3))
                .copy_from(&(identity * spectral * dt.powi(2) / 2.0));
            q_noise
                .view_mut((rate_col, value_col), (3, 3))
                .copy_from(&(identity * spectral * dt.powi(2) / 2.0));
            q_noise
                .view_mut((rate_col, rate_col), (3, 3))
                .copy_from(&(identity * spectral * dt));
        }
        self.p_cov = &f * &self.p_cov * f.transpose() + q_noise;
    }

    /// One step of dead reckoning from IMU `imu`'s sample, using that IMU's bias and noise.
    fn predict_inertial(&mut self, dt: f64, imu: usize, gyro: &Vector3<f64>, accel: &Vector3<f64>) {
        let dim = self.dim();
        let layout = self.layout;
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
        f.view_mut((3, layout.accel_bias_col(imu)), (3, 3))
            .copy_from(&(-rotation.matrix() * dt));
        f.view_mut((6, 6), (3, 3))
            .copy_from(&(Matrix3::identity() - skew(&unbiased_gyro) * dt));
        f.view_mut((6, layout.gyro_bias_col(imu)), (3, 3))
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
            blocks.push((layout.gyro_bias_col(other), noise.gyro_random_walk));
            blocks.push((layout.accel_bias_col(other), noise.accel_random_walk));
        }
        for (block, density) in blocks {
            q_noise
                .view_mut((block, block), (3, 3))
                .copy_from(&(Matrix3::identity() * density * density * dt));
        }
        self.p_cov = &f * &self.p_cov * f.transpose() + q_noise;
    }

    /// Projects out what a planar body cannot do. A hard constraint rather than pseudo-
    /// measurements: those leave the state off the plane between updates and P ill-conditioned.
    fn constrain(&mut self) {
        if !self.planar {
            return;
        }
        self.x.position.z = 0.0;
        self.x.velocity.z = 0.0;
        self.x.q = yaw_only(&self.x.q);
        if self.layout.angular_velocity_col().is_some() {
            self.x.angular_velocity.x = 0.0;
            self.x.angular_velocity.y = 0.0;
        }
        for clone in self.x.clones.iter_mut().flatten() {
            clone.position.z = 0.0;
            clone.q = yaw_only(&clone.q);
        }
        for col in self.layout.pinned_cols() {
            self.p_cov.row_mut(col).fill(0.0);
            self.p_cov.column_mut(col).fill(0.0);
        }
    }

    /// Copies the current pose, with its correlations, into clone `slot`.
    pub fn clone_pose(&mut self, slot: usize) {
        self.x.clones[slot] = Some(PoseClone {
            position: self.x.position,
            q: self.x.q,
        });
        let dim = self.dim();
        let position = self.layout.clone_position_col(slot);
        let rotation = self.layout.clone_rotation_col(slot);
        let mut selector = Cov::identity(dim, dim);
        for offset in 0..3 {
            selector.row_mut(position + offset).fill(0.0);
            selector.row_mut(rotation + offset).fill(0.0);
            selector[(position + offset, offset)] = 1.0;
            selector[(rotation + offset, 6 + offset)] = 1.0;
        }
        self.p_cov = &selector * &self.p_cov * selector.transpose();
    }

    /// The innovation covariance's inverse, or None when `gate` rejects the measurement: its
    /// squared Mahalanobis distance under the full innovation covariance, per measurement
    /// dimension, at or past `gate`. 0 disables the gate.
    // The negated comparison is deliberate: a NaN distance must reject.
    #[allow(clippy::neg_cmp_op_on_partial_ord)]
    fn gated_innovation_inverse(
        &self,
        residual: &DVector<f64>,
        jacobian: &Jacobian,
        variance: &DVector<f64>,
        gate: f64,
    ) -> Option<Cov> {
        let innovation =
            jacobian * &self.p_cov * jacobian.transpose() + Cov::from_diagonal(variance);
        let innovation_inverse = innovation.try_inverse()?;
        if gate > 0.0 {
            let mahalanobis_sq = (residual.transpose() * &innovation_inverse * residual)[(0, 0)];
            if !(mahalanobis_sq < gate * residual.len() as f64) {
                return None;
            }
        }
        Some(innovation_inverse)
    }

    pub fn within_gate(
        &self,
        residual: &DVector<f64>,
        jacobian: &Jacobian,
        variance: &DVector<f64>,
        gate: f64,
    ) -> bool {
        self.gated_innovation_inverse(residual, jacobian, variance, gate)
            .is_some()
    }

    /// False, leaving the state untouched, when the gate rejects the measurement.
    pub fn update(
        &mut self,
        residual: &DVector<f64>,
        jacobian: &Jacobian,
        variance: &DVector<f64>,
        gate: f64,
    ) -> bool {
        let Some(innovation_inverse) =
            self.gated_innovation_inverse(residual, jacobian, variance, gate)
        else {
            return false;
        };
        let noise = Cov::from_diagonal(variance);
        let kalman_gain = &self.p_cov * jacobian.transpose() * &innovation_inverse;
        let dx: DVector<f64> = &kalman_gain * residual;
        let identity_minus_kh = Cov::identity(self.dim(), self.dim()) - &kalman_gain * jacobian;
        self.p_cov = &identity_minus_kh * &self.p_cov * identity_minus_kh.transpose()
            + &kalman_gain * noise * kalman_gain.transpose();
        self.inject(&dx);
        true
    }

    fn inject(&mut self, dx: &DVector<f64>) {
        let layout = self.layout;
        self.x.position += dx.fixed_rows::<3>(0);
        self.x.velocity += dx.fixed_rows::<3>(3);
        self.x.q *= UnitQuaternion::from_scaled_axis(dx.fixed_rows::<3>(6).into_owned());
        if let Some(col) = layout.angular_velocity_col() {
            self.x.angular_velocity += dx.fixed_rows::<3>(col);
        }
        for (imu, bias) in self.x.biases.iter_mut().enumerate() {
            bias.gyro += dx.fixed_rows::<3>(layout.gyro_bias_col(imu));
            bias.accel += dx.fixed_rows::<3>(layout.accel_bias_col(imu));
        }
        for (slot, clone) in self.x.clones.iter_mut().enumerate() {
            if let Some(clone) = clone {
                clone.position += dx.fixed_rows::<3>(layout.clone_position_col(slot));
                clone.q *= UnitQuaternion::from_scaled_axis(
                    dx.fixed_rows::<3>(layout.clone_rotation_col(slot))
                        .into_owned(),
                );
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const GRAVITY: f64 = 9.80665;
    const MS: i64 = 1_000_000;

    fn stds() -> InitialStds {
        InitialStds {
            position: 0.1,
            velocity: 0.1,
            rotation: 0.05,
            angular_velocity: 0.1,
            bias: 0.01,
        }
    }

    fn imu_noise() -> Noise {
        Noise {
            gyro_noise_density: 0.01,
            gyro_random_walk: 0.001,
            accel_noise_density: 0.1,
            accel_random_walk: 0.01,
        }
    }

    fn still_input(imu: usize, gyro: Vector3<f64>) -> Option<ImuInput> {
        Some(ImuInput {
            imu,
            gyro,
            accel: Vector3::new(0.0, 0.0, GRAVITY),
        })
    }

    fn level_filter() -> Filter {
        let mut filter = Filter {
            noise: vec![imu_noise()],
            gravity: GRAVITY,
            ..Filter::default()
        };
        filter.init(
            0,
            UnitQuaternion::identity(),
            vec![ImuBias::default()],
            still_input(0, Vector3::zeros()),
            &stds(),
        );
        filter
    }

    fn constant_velocity_filter(clones: usize) -> Filter {
        let mut filter = Filter {
            layout: Layout { imus: 0, clones },
            process_noise: ProcessNoise {
                linear_acceleration: 0.5,
                angular_acceleration: 0.5,
            },
            ..Filter::default()
        };
        filter.init(0, UnitQuaternion::identity(), Vec::new(), None, &stds());
        filter
    }

    fn position_jacobian(rows: usize, dim: usize) -> Jacobian {
        let mut jacobian = Jacobian::zeros(rows, dim);
        jacobian.view_mut((0, 0), (3, 3)).fill_with_identity();
        jacobian
    }

    /// Steps of `step_ms`, `steps` of them, from wherever the filter stands.
    fn advance(filter: &mut Filter, steps: i64, step_ms: i64) {
        let start = filter.x.time_ns;
        for step in 1..=steps {
            filter.predict_to(start + step * step_ms * MS);
        }
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
    fn the_layout_places_the_rate_state_only_without_an_imu() {
        let without = Layout { imus: 0, clones: 1 };
        assert_eq!(without.angular_velocity_col(), Some(9));
        assert_eq!(without.clone_position_col(0), 12);
        assert_eq!(without.dim(), 18);
        let with = Layout { imus: 2, clones: 1 };
        assert_eq!(with.angular_velocity_col(), None);
        assert_eq!(with.accel_bias_col(1), 18);
        assert_eq!(with.clone_rotation_col(0), 24);
        assert_eq!(with.dim(), 27);
    }

    #[test]
    fn stationary_propagation_holds_the_state() {
        let mut filter = level_filter();
        advance(&mut filter, 100, 10);
        assert!(filter.x.position.norm() < 1e-9);
        assert!(filter.x.velocity.norm() < 1e-9);
    }

    #[test]
    fn inertial_propagation_integrates_constant_velocity() {
        let mut filter = level_filter();
        filter.x.velocity = Vector3::new(1.0, 0.0, 0.0);
        advance(&mut filter, 100, 10);
        assert!((filter.x.position.x - 1.0).abs() < 1e-9);
    }

    #[test]
    fn an_imu_gap_is_not_bridged_by_one_held_sample() {
        let mut filter = level_filter();
        filter.x.input.as_mut().expect("held").accel.x = 1.0;
        assert!(!filter.predict_to(2_000 * MS));
        assert_eq!(filter.x.position, Vector3::zeros());
    }

    #[test]
    fn the_constant_velocity_model_carries_position_and_heading() {
        let mut filter = constant_velocity_filter(0);
        filter.x.velocity = Vector3::new(1.0, 0.0, 0.0);
        filter.x.angular_velocity = Vector3::new(0.0, 0.0, 0.5);
        let before = filter.p_cov.trace();
        advance(&mut filter, 100, 10);
        assert!((filter.x.position.x - 1.0).abs() < 1e-9);
        assert!((filter.x.q.euler_angles().2 - 0.5).abs() < 1e-9);
        assert!(filter.p_cov.trace() > before);
    }

    #[test]
    fn constant_velocity_uncertainty_grows_with_the_cube_of_time_in_position() {
        let mut filter = constant_velocity_filter(0);
        filter.p_cov.fill(0.0);
        advance(&mut filter, 1, 2_000);
        // q dt^3 / 3 with q = 0.25, dt = 2.
        assert!((filter.p_cov[(0, 0)] - 0.25 * 8.0 / 3.0).abs() < 1e-12);
        assert!((filter.p_cov[(3, 3)] - 0.5).abs() < 1e-12);
    }

    #[test]
    fn planar_pins_z_roll_pitch_and_their_rates() {
        let mut filter = constant_velocity_filter(0);
        filter.planar = true;
        filter.x.velocity = Vector3::new(1.0, 0.0, 0.3);
        filter.x.angular_velocity = Vector3::new(0.2, -0.1, 0.4);
        advance(&mut filter, 10, 10);
        assert_eq!(filter.x.position.z, 0.0);
        let (roll, pitch, yaw) = filter.x.q.euler_angles();
        assert!(roll.abs() < 1e-12 && pitch.abs() < 1e-12);
        assert!(yaw > 0.0);
        for col in [2, 5, 6, 7, 9, 10] {
            assert_eq!(filter.p_cov.row(col).norm(), 0.0, "col {col}");
        }
    }

    #[test]
    fn a_position_update_pulls_the_estimate() {
        let mut filter = level_filter();
        advance(&mut filter, 1, 10);
        let residual = DVector::from_vec(vec![1.0, 0.0, 0.0]);
        let variance = DVector::from_element(3, 1e-4);
        assert!(filter.update(
            &residual,
            &position_jacobian(3, filter.dim()),
            &variance,
            0.0
        ));
        assert!(filter.x.position.x > 0.9);
    }

    #[test]
    fn position_updates_correct_velocity_through_the_cross_covariance() {
        let mut filter = level_filter();
        filter.x.velocity = Vector3::new(1.0, 0.0, 0.0); // true velocity is zero; the filter is wrong
        for _ in 0..50 {
            advance(&mut filter, 1, 10);
            let residual = -DVector::from_column_slice(filter.x.position.as_slice());
            let variance = DVector::from_element(3, 1e-6);
            filter.update(
                &residual,
                &position_jacobian(3, filter.dim()),
                &variance,
                0.0,
            );
        }
        assert!(filter.x.velocity.x.abs() < 0.1);
    }

    #[test]
    fn the_mahalanobis_gate_rejects_an_outlier_and_passes_an_inlier() {
        let mut filter = level_filter();
        advance(&mut filter, 1, 10);
        let variance = DVector::from_element(3, 1e-4);
        let outlier = DVector::from_vec(vec![100.0, 0.0, 0.0]);
        assert!(!filter.update(
            &outlier,
            &position_jacobian(3, filter.dim()),
            &variance,
            9.0
        ));
        let inlier = DVector::from_vec(vec![0.01, 0.0, 0.0]);
        assert!(filter.update(&inlier, &position_jacobian(3, filter.dim()), &variance, 9.0));
    }

    #[test]
    fn a_gate_of_zero_disables_rejection() {
        let mut filter = level_filter();
        let variance = DVector::from_element(3, 1e-4);
        let outlier = DVector::from_vec(vec![100.0, 0.0, 0.0]);
        assert!(filter.update(
            &outlier,
            &position_jacobian(3, filter.dim()),
            &variance,
            0.0
        ));
    }

    #[test]
    fn a_gyro_bias_propagates_into_a_tilt() {
        let mut filter = level_filter();
        filter.x.input = still_input(0, Vector3::new(0.02, 0.0, 0.0));
        advance(&mut filter, 100, 10);
        let tilt = filter.x.q.scaled_axis();
        assert!((tilt.x - 0.02).abs() < 1e-3);
    }

    #[test]
    fn an_angular_rate_update_observes_the_held_imus_gyro_bias() {
        let mut filter = level_filter();
        filter.x.input = still_input(0, Vector3::new(0.05, 0.0, 0.0));
        for _ in 0..50 {
            advance(&mut filter, 1, 10);
            // A perfect rate sensor says the body is not rotating.
            let (predicted, correction) = filter.angular_velocity();
            let (col, sign) = correction.expect("an imu sample is held");
            let residual = -DVector::from_column_slice(predicted.as_slice());
            let mut jacobian = Jacobian::zeros(3, filter.dim());
            jacobian
                .view_mut((0, col), (3, 3))
                .copy_from(&(Matrix3::identity() * sign));
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
            0,
            UnitQuaternion::identity(),
            vec![ImuBias {
                gyro: Vector3::new(0.1, 0.0, 0.0),
                accel: Vector3::zeros(),
            }],
            None,
            &InitialStds {
                position: 2.0,
                velocity: 3.0,
                rotation: 0.5,
                angular_velocity: 7.0,
                bias: 0.01,
            },
        );
        assert!((filter.p_cov[(0, 0)] - 4.0).abs() < 1e-12);
        assert!((filter.p_cov[(3, 3)] - 9.0).abs() < 1e-12);
        assert!((filter.p_cov[(6, 6)] - 0.25).abs() < 1e-12);
        assert!((filter.p_cov[(9, 9)] - 1e-4).abs() < 1e-12);
        assert!((filter.x.biases[0].gyro.x - 0.1).abs() < 1e-12);
    }

    #[test]
    fn two_imus_keep_separate_bias_states() {
        let mut filter = Filter {
            noise: vec![imu_noise(), imu_noise()],
            gravity: GRAVITY,
            ..Filter::default()
        };
        filter.init(
            0,
            UnitQuaternion::identity(),
            vec![ImuBias::default(), ImuBias::default()],
            None,
            &stds(),
        );
        assert_eq!(filter.dim(), 21);
        // Only the second IMU's gyro is biased; a rate update on its block finds it there.
        let gyro_reading = Vector3::new(0.05, 0.0, 0.0);
        for step in 0..100 {
            let imu = step % 2;
            let gyro = if imu == 1 {
                gyro_reading
            } else {
                Vector3::zeros()
            };
            filter.x.input = still_input(imu, gyro);
            advance(&mut filter, 1, 5);
            if imu == 1 {
                let predicted = gyro_reading - filter.x.biases[1].gyro;
                let residual = -DVector::from_column_slice(predicted.as_slice());
                let mut jacobian = Jacobian::zeros(3, filter.dim());
                jacobian
                    .view_mut((0, filter.layout.gyro_bias_col(1)), (3, 3))
                    .copy_from(&(-Matrix3::identity()));
                let variance = DVector::from_element(3, 1e-6);
                filter.update(&residual, &jacobian, &variance, 0.0);
            }
        }
        assert!((filter.x.biases[1].gyro.x - 0.05).abs() < 2e-3);
        assert!(filter.x.biases[0].gyro.x.abs() < 2e-2);
    }

    #[test]
    fn a_clone_carries_the_poses_covariance_and_correlation() {
        let mut filter = constant_velocity_filter(1);
        filter.x.velocity = Vector3::new(1.0, 0.0, 0.0);
        advance(&mut filter, 10, 10);
        filter.clone_pose(0);
        let col = filter.layout.clone_position_col(0);
        assert_eq!(filter.p_cov[(col, col)], filter.p_cov[(0, 0)]);
        assert_eq!(filter.p_cov[(col, 0)], filter.p_cov[(0, 0)]);
        advance(&mut filter, 10, 10);
        // The clone stands still while the pose moves on and loosens.
        assert!((filter.x.clones[0].as_ref().expect("cloned").position.x - 0.1).abs() < 1e-12);
        assert!(filter.p_cov[(0, 0)] > filter.p_cov[(col, col)]);
    }
}
