// Copyright 2026 Dimensional Inc.
// SPDX-License-Identifier: Apache-2.0
//
// Point-LIO's estimator, mirrored so the two can be compared stage by stage, bit for bit: the
// error-state filter over a 30-d state in which the IMU is a measurement of the body's angular
// velocity and specific force rather than its input. Each expression keeps Point-LIO's order
// of operations, since floating point is not associative; the quirks go once the outputs
// agree. Not to leave this branch while it mirrors (GPL upstream).

use nalgebra::{Const, DMatrix, DVector, Dyn, Matrix3, OMatrix, SMatrix, SVector, Vector3};

use super::plane_fit::{fit_plane, PLANE_POINTS};
use super::point_map::{to_world, PointMap};
use super::scan_points::SurfacePoint;

/// The 99% quantiles of a chi-squared distribution with 1 to 6 degrees of freedom.
const CHI_SQUARED_99: [f64; 6] = [6.635, 9.210, 11.345, 13.277, 15.086, 16.812];

pub const STATE_DIM: usize = 30;
pub type StateVector = SVector<f64, STATE_DIM>;
pub type StateMatrix = SMatrix<f64, STATE_DIM, STATE_DIM>;

/// Error-state layout: position 0, rotation 3, lidar mount rotation 6, lidar mount translation
/// 9, velocity 12, angular velocity 15, specific force 18, gravity 21, gyro bias 24, accel
/// bias 27.
#[derive(Clone, Debug, PartialEq)]
pub struct LioState {
    pub position: Vector3<f64>,
    pub rotation: Matrix3<f64>,
    pub mount_rotation: Matrix3<f64>,
    pub mount_translation: Vector3<f64>,
    pub velocity: Vector3<f64>,
    pub angular_velocity: Vector3<f64>,
    pub specific_force: Vector3<f64>,
    pub gravity: Vector3<f64>,
    pub gyro_bias: Vector3<f64>,
    pub accel_bias: Vector3<f64>,
}

impl Default for LioState {
    fn default() -> Self {
        Self {
            position: Vector3::zeros(),
            rotation: Matrix3::identity(),
            mount_rotation: Matrix3::identity(),
            mount_translation: Vector3::zeros(),
            velocity: Vector3::zeros(),
            angular_velocity: Vector3::zeros(),
            specific_force: Vector3::zeros(),
            gravity: Vector3::zeros(),
            gyro_bias: Vector3::zeros(),
            accel_bias: Vector3::zeros(),
        }
    }
}

fn block(v: &StateVector, at: usize) -> Vector3<f64> {
    v.fixed_rows::<3>(at).into_owned()
}

fn set_block(v: &mut StateVector, at: usize, value: &Vector3<f64>) {
    v.fixed_rows_mut::<3>(at).copy_from(value);
}

impl LioState {
    /// Moves along `step` scaled: vectors add, rotations compose on the right.
    pub fn advance(&mut self, step: &StateVector, scale: f64) {
        self.position += scale * block(step, 0);
        self.rotation *= exp(&(block(step, 3) * scale));
        self.mount_rotation *= exp(&(block(step, 6) * scale));
        self.mount_translation += scale * block(step, 9);
        self.velocity += scale * block(step, 12);
        self.angular_velocity += scale * block(step, 15);
        self.specific_force += scale * block(step, 18);
        self.gravity += scale * block(step, 21);
        self.gyro_bias += scale * block(step, 24);
        self.accel_bias += scale * block(step, 27);
    }

    /// The step from `other` to `self`.
    pub fn difference(&self, other: &Self) -> StateVector {
        let mut d = StateVector::zeros();
        set_block(&mut d, 0, &(self.position - other.position));
        set_block(
            &mut d,
            3,
            &log(&(other.rotation.transpose() * self.rotation)),
        );
        set_block(
            &mut d,
            6,
            &log(&(other.mount_rotation.transpose() * self.mount_rotation)),
        );
        set_block(
            &mut d,
            9,
            &(self.mount_translation - other.mount_translation),
        );
        set_block(&mut d, 12, &(self.velocity - other.velocity));
        set_block(
            &mut d,
            15,
            &(self.angular_velocity - other.angular_velocity),
        );
        set_block(&mut d, 18, &(self.specific_force - other.specific_force));
        set_block(&mut d, 21, &(self.gravity - other.gravity));
        set_block(&mut d, 24, &(self.gyro_bias - other.gyro_bias));
        set_block(&mut d, 27, &(self.accel_bias - other.accel_bias));
        d
    }
}

pub fn hat(v: &Vector3<f64>) -> Matrix3<f64> {
    Matrix3::new(0.0, -v[2], v[1], v[2], 0.0, -v[0], -v[1], v[0], 0.0)
}

/// A 3x3 product summed in a fixed order, a(i,0)b(0,j) + a(i,1)b(1,j) + a(i,2)b(2,j).
fn product(a: &Matrix3<f64>, b: &Matrix3<f64>) -> Matrix3<f64> {
    let mut r = Matrix3::zeros();
    for i in 0..3 {
        for j in 0..3 {
            r[(i, j)] = a[(i, 0)] * b[(0, j)] + a[(i, 1)] * b[(1, j)] + a[(i, 2)] * b[(2, j)];
        }
    }
    r
}

/// |v|, summed as v0^2 + (v1^2 + v2^2).
pub fn norm(v: &Vector3<f64>) -> f64 {
    (v[0] * v[0] + (v[1] * v[1] + v[2] * v[2])).sqrt()
}

fn rodrigues(k: &Matrix3<f64>, angle: f64) -> Matrix3<f64> {
    let s = angle.sin();
    let kk = product(&((1.0 - angle.cos()) * k), k);
    let mut r = Matrix3::identity();
    for i in 0..3 {
        for j in 0..3 {
            r[(i, j)] = r[(i, j)] + s * k[(i, j)] + kk[(i, j)];
        }
    }
    r
}

/// The rotation of axis-angle `v`; identity below 1e-7 rad.
pub fn exp(v: &Vector3<f64>) -> Matrix3<f64> {
    let angle = norm(v);
    if angle > 0.0000001 {
        rodrigues(&hat(&(v / angle)), angle)
    } else {
        Matrix3::identity()
    }
}

/// The axis-angle of `r`.
pub fn log(r: &Matrix3<f64>) -> Vector3<f64> {
    let trace = r[(0, 0)] + (r[(1, 1)] + r[(2, 2)]);
    let theta = if trace > 3.0 - 1e-6 {
        0.0
    } else {
        (0.5 * (trace - 1.0)).acos()
    };
    let k = Vector3::new(
        r[(2, 1)] - r[(1, 2)],
        r[(0, 2)] - r[(2, 0)],
        r[(1, 0)] - r[(0, 1)],
    );
    if theta.abs() < 0.001 {
        0.5 * k
    } else {
        (0.5 * theta / theta.sin()) * k
    }
}

/// SO(3)'s left Jacobian at `v`; identity below 1e-11 rad.
pub fn left_jacobian(v: &Vector3<f64>) -> Matrix3<f64> {
    let sq = v[0] * v[0] + v[1] * v[1] + v[2] * v[2];
    let angle = sq.sqrt();
    if angle < 1e-11 {
        return Matrix3::identity();
    }
    let h = hat(v);
    let c1 = (1.0 - angle.cos()) / sq;
    let c2 = (1.0 - angle.sin() / angle) / sq;
    let hh = product(&(c2 * h), &h);
    let mut r = Matrix3::identity();
    for i in 0..3 {
        for j in 0..3 {
            r[(i, j)] = r[(i, j)] + c1 * h[(i, j)] + hh[(i, j)];
        }
    }
    r
}

/// The state's rates: position by velocity, rotation by angular velocity, velocity by the
/// specific force turned into the world plus gravity; the rest hold.
pub fn rates(s: &LioState) -> StateVector {
    let mut r = StateVector::zeros();
    let inertial = s.rotation * s.specific_force;
    for i in 0..3 {
        r[i] = s.velocity[i];
        r[i + 3] = s.angular_velocity[i];
        r[i + 12] = inertial[i] + s.gravity[i];
    }
    r
}

/// d rates / d error state.
pub fn rates_jacobian(s: &LioState) -> StateMatrix {
    let mut j = StateMatrix::zeros();
    j.fixed_view_mut::<3, 3>(0, 12)
        .copy_from(&Matrix3::identity());
    j.fixed_view_mut::<3, 3>(12, 3)
        .copy_from(&(-s.rotation * hat(&s.specific_force)));
    j.fixed_view_mut::<3, 3>(12, 18).copy_from(&s.rotation);
    j.fixed_view_mut::<3, 3>(12, 21)
        .copy_from(&Matrix3::identity());
    j.fixed_view_mut::<3, 3>(3, 15)
        .copy_from(&Matrix3::identity());
    j
}

/// Process noise: random walks on velocity, angular velocity, specific force and the biases.
pub fn process_noise(
    velocity: f64,
    angular_velocity: f64,
    specific_force: f64,
    gyro_bias: f64,
    accel_bias: f64,
) -> StateMatrix {
    let mut q = StateMatrix::zeros();
    for i in 0..3 {
        q[(12 + i, 12 + i)] = velocity;
        q[(15 + i, 15 + i)] = angular_velocity;
        q[(18 + i, 18 + i)] = specific_force;
        q[(24 + i, 24 + i)] = gyro_bias;
        q[(27 + i, 27 + i)] = accel_bias;
    }
    q
}

/// The covariance the filter starts from.
pub fn initial_covariance() -> StateMatrix {
    let mut p = StateMatrix::identity() * 0.01;
    for i in 0..3 {
        p[(21 + i, 21 + i)] = 0.0001;
    }
    for i in 0..6 {
        p[(24 + i, 24 + i)] = 0.001;
    }
    p
}

/// Point-LIO's 30 states, plus states it does not have (each further IMU's gyro then accel
/// bias) carried jointly with them: `p_extra` is their covariance with the 30, `p_extra_extra`
/// among themselves. With none, every operation is Point-LIO's own.
#[derive(Clone, Debug)]
pub struct LioFilter {
    pub x: LioState,
    pub p: StateMatrix,
    pub extra: DVector<f64>,
    pub p_extra: OMatrix<f64, Const<STATE_DIM>, Dyn>,
    pub p_extra_extra: DMatrix<f64>,
    /// Random-walk variances of the extra states, scaled by dt^2 as Point-LIO scales its own.
    pub q_extra: DVector<f64>,
}

impl Default for LioFilter {
    fn default() -> Self {
        Self {
            x: LioState::default(),
            p: StateMatrix::identity(),
            extra: DVector::zeros(0),
            p_extra: OMatrix::<f64, Const<STATE_DIM>, Dyn>::zeros(0),
            p_extra_extra: DMatrix::zeros(0, 0),
            q_extra: DVector::zeros(0),
        }
    }
}

impl LioFilter {
    /// Moves the state `dt` along its rates (`state`) and/or grows the covariance over `dt`
    /// (`covariance`), the two halves Point-LIO runs at different times.
    pub fn predict(&mut self, dt: f64, q: &StateMatrix, state: bool, covariance: bool) {
        if state {
            let r = rates(&self.x);
            self.x.advance(&r, dt);
        }
        if covariance {
            let r = rates(&self.x);
            let j = rates_jacobian(&self.x);
            let mut transition_rates = j;
            let mut transition = StateMatrix::identity();
            for at in [3usize, 6] {
                let mut turn = Vector3::zeros();
                for i in 0..3 {
                    turn[i] = -r[at + i] * dt;
                }
                transition
                    .fixed_view_mut::<3, 3>(at, at)
                    .copy_from(&exp(&turn));
                let jacobian = left_jacobian(&turn);
                for column in 0..STATE_DIM {
                    let mapped = jacobian * j.fixed_view::<3, 1>(at, column);
                    transition_rates
                        .fixed_view_mut::<3, 1>(at, column)
                        .copy_from(&mapped);
                }
            }
            transition += transition_rates * dt;
            self.p = transition * self.p * transition.transpose() + q * (dt * dt);
            if !self.extra.is_empty() {
                self.p_extra = transition * &self.p_extra;
                for i in 0..self.extra.len() {
                    self.p_extra_extra[(i, i)] += self.q_extra[i] * (dt * dt);
                }
            }
        }
    }

    /// Adds extra states, independent of the rest, with these values and variances and
    /// random-walk variances.
    pub fn add_extra(&mut self, values: &[f64], variances: &[f64], random_walks: &[f64]) {
        let (n, added) = (self.extra.len(), values.len());
        self.extra = self.extra.clone().insert_rows(n, added, 0.0);
        self.q_extra = self.q_extra.clone().insert_rows(n, added, 0.0);
        self.p_extra = self.p_extra.clone().insert_columns(n, added, 0.0);
        self.p_extra_extra = self
            .p_extra_extra
            .clone()
            .insert_rows(n, added, 0.0)
            .insert_columns(n, added, 0.0);
        for i in 0..added {
            self.extra[n + i] = values[i];
            self.q_extra[n + i] = random_walks[i];
            self.p_extra_extra[(n + i, n + i)] = variances[i];
        }
    }

    /// After an update that observed only the 30 states (from `before`, `before_p`), carries
    /// its effect to the extra states through their covariance with the 30: the same as had
    /// the update been made over all of them, since the gain on an unobserved state is its
    /// regression on the observed ones times theirs.
    pub fn carry_extra(&mut self, before: &LioState, before_p: &StateMatrix) {
        if self.extra.is_empty() || self.p == *before_p {
            return;
        }
        // P^-1 P_ce, by a Cholesky solve rather than an inverse: this runs after every point group.
        let Some(cholesky) = before_p.cholesky() else {
            return;
        };
        let solved = cholesky.solve(&self.p_extra);
        let step = self.x.difference(before);
        self.extra += solved.transpose() * step;
        self.p_extra_extra -= solved.transpose() * ((before_p - self.p) * &solved);
        self.p_extra = self.p * solved;
    }
}

/// How an IMU sample is weighed as a measurement.
#[derive(Clone, Copy, Debug)]
pub struct ImuMeasurement {
    /// Variances of one gyro (rad^2/s^2) and one accelerometer (m^2/s^4) reading.
    pub gyro_variance: f64,
    pub accel_variance: f64,
    /// Readings within 1% of these are clipped and say nothing (rad/s, and the
    /// accelerometer's own unit); `None` trusts every reading.
    pub clip: Option<(f64, f64)>,
    /// m/s^2 per accelerometer unit.
    pub accel_scale: f64,
}

impl LioFilter {
    /// One IMU sample (`gyro` rad/s, `accel` in the accelerometer's unit) as a measurement of
    /// the angular velocity and specific force, each read with its bias added.
    pub fn update_imu(&mut self, gyro: &Vector3<f64>, accel: &Vector3<f64>, m: &ImuMeasurement) {
        let gyro_residual = gyro - self.x.angular_velocity - self.x.gyro_bias;
        let accel_residual = accel * m.accel_scale - self.x.specific_force - self.x.accel_bias;
        let mut residual = SVector::<f64, 6>::zeros();
        residual.fixed_rows_mut::<3>(0).copy_from(&gyro_residual);
        residual.fixed_rows_mut::<3>(3).copy_from(&accel_residual);
        let mut clipped = [false; 6];
        if let Some((gyro_clip, accel_clip)) = m.clip {
            for axis in 0..3 {
                if gyro[axis].abs() >= 0.99 * gyro_clip {
                    clipped[axis] = true;
                    residual[axis] = 0.0;
                }
            }
            for axis in 0..3 {
                if accel[axis].abs() >= 0.99 * accel_clip {
                    clipped[3 + axis] = true;
                    residual[3 + axis] = 0.0;
                }
            }
        }
        // Each reading is a rate state plus its bias: row l picks columns 15 + l and 24 + l.
        let mut p_ht = SMatrix::<f64, STATE_DIM, 6>::zeros();
        let mut h_p = SMatrix::<f64, 6, STATE_DIM>::zeros();
        for l in (0..6).filter(|&l| !clipped[l]) {
            p_ht.set_column(l, &(self.p.column(15 + l) + self.p.column(24 + l)));
            h_p.set_row(l, &(self.p.row(15 + l) + self.p.row(24 + l)));
        }
        let mut innovation = SMatrix::<f64, 6, 6>::zeros();
        for l in 0..6 {
            if !clipped[l] {
                innovation.set_column(l, &(h_p.column(15 + l) + h_p.column(24 + l)));
            }
            innovation[(l, l)] += if l < 3 {
                m.gyro_variance
            } else {
                m.accel_variance
            };
        }
        let gain = p_ht * innovation.try_inverse().expect("IMU innovation singular");
        let step = gain * residual;
        self.p -= gain * h_p;
        self.x.advance(&step, 1.0);
    }
}

/// Where an IMU sits on the body and what the body's rotation is doing there.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ImuMount {
    /// body_from_imu.
    pub rotation: Matrix3<f64>,
    /// m: the IMU's origin in the body frame.
    pub lever: Vector3<f64>,
    /// rad/s^2, body frame: the angular acceleration, which turns into linear acceleration
    /// away from the body's origin, and its variance per axis.
    pub angular_acceleration: Vector3<f64>,
    pub angular_acceleration_variance: f64,
}

impl ImuMount {
    /// The IMU is the body: Point-LIO's own case.
    pub fn is_body(&self) -> bool {
        self.rotation == Matrix3::identity() && self.lever == Vector3::zeros()
    }
}

/// A measurement's residual against its predicted spread with the variances it was given:
/// r^T S^-1 r over `dof` rows, and whether the update took it.
#[derive(Clone, Copy, Debug)]
pub struct Innovation {
    pub squared: f64,
    pub dof: usize,
    pub accepted: bool,
}

/// Medians of a chi-squared distribution with 1 to 6 degrees of freedom: what `squared`
/// sits at, half the time, when the given variances are right.
pub const CHI_SQUARED_MEDIAN: [f64; 6] = [0.455, 1.386, 2.366, 3.357, 4.351, 5.348];

/// Rows over the 30 states, one per measured axis.
pub type CoreRows = OMatrix<f64, Dyn, Const<STATE_DIM>>;

/// Where a reading's bias lives: the 30 states' own (the first IMU), or the extra states at
/// this index (gyro, then accel).
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum BiasAt {
    Core,
    Extra(usize),
}

impl LioFilter {
    /// An IMU sample (`gyro` rad/s, `accel` in the accelerometer's unit) at `mount`: its gyro
    /// reads the body's angular velocity and its accelerometer the specific force at its own
    /// origin, both turned into its axes, each with its bias added.
    pub fn update_mounted_imu(
        &mut self,
        gyro: &Vector3<f64>,
        accel: &Vector3<f64>,
        m: &ImuMeasurement,
        mount: &ImuMount,
        bias: BiasAt,
    ) {
        if bias == BiasAt::Core && mount.is_body() {
            return self.update_imu(gyro, accel, m);
        }
        let x = &self.x;
        let to_imu = mount.rotation.transpose();
        let r = mount.lever;
        let w = x.angular_velocity;
        let at_lever = w.cross(&w.cross(&r)) + mount.angular_acceleration.cross(&r);
        let mut predicted = SVector::<f64, 6>::zeros();
        predicted.fixed_rows_mut::<3>(0).copy_from(&(to_imu * w));
        predicted
            .fixed_rows_mut::<3>(3)
            .copy_from(&(to_imu * (x.specific_force + at_lever)));
        let mut h = CoreRows::zeros(6);
        h.fixed_view_mut::<3, 3>(0, 15).copy_from(&to_imu);
        h.fixed_view_mut::<3, 3>(3, 18).copy_from(&to_imu);
        // d(w x (w x r))/dw = (w.r) I + w r^T - 2 r w^T.
        let centripetal =
            Matrix3::identity() * w.dot(&r) + w * r.transpose() - 2.0 * r * w.transpose();
        h.fixed_view_mut::<3, 3>(3, 15)
            .copy_from(&(to_imu * centripetal));
        let (bias_gyro, bias_accel) = match bias {
            BiasAt::Core => {
                h.fixed_view_mut::<3, 3>(0, 24)
                    .copy_from(&Matrix3::identity());
                h.fixed_view_mut::<3, 3>(3, 27)
                    .copy_from(&Matrix3::identity());
                (x.gyro_bias, x.accel_bias)
            }
            BiasAt::Extra(at) => (
                self.extra.fixed_rows::<3>(at).into_owned(),
                self.extra.fixed_rows::<3>(at + 3).into_owned(),
            ),
        };
        let mut residual = DVector::zeros(6);
        let mut variances = DVector::zeros(6);
        for axis in 0..3 {
            residual[axis] = gyro[axis] - predicted[axis] - bias_gyro[axis];
            residual[3 + axis] =
                accel[axis] * m.accel_scale - predicted[3 + axis] - bias_accel[axis];
            variances[axis] = m.gyro_variance;
            variances[3 + axis] =
                m.accel_variance + mount.angular_acceleration_variance * r.norm_squared();
        }
        if let Some((gyro_clip, accel_clip)) = m.clip {
            for axis in 0..3 {
                for (row, reading, clip) in [
                    (axis, gyro[axis], gyro_clip),
                    (3 + axis, accel[axis], accel_clip),
                ] {
                    if reading.abs() >= 0.99 * clip {
                        h.row_mut(row).fill(0.0);
                        residual[row] = 0.0;
                    }
                }
            }
        }
        match bias {
            BiasAt::Core => {
                self.update_rows(&h, &residual, &variances, 1.0, false);
            }
            BiasAt::Extra(at) => {
                let mut h_extra = DMatrix::zeros(6, self.extra.len());
                for row in 0..6 {
                    if h.row(row).iter().any(|v| *v != 0.0) {
                        h_extra[(row, at + row)] = 1.0;
                    }
                }
                self.update_joint(&h, &h_extra, &residual, &variances);
            }
        }
    }

    /// Rows observing the 30 states only: `residual` and per-row `variances`, which the update
    /// uses scaled by `scale`. With `gate`, the state is left untouched when the residual is
    /// beyond the 99% bound of its predicted spread. Joseph form: the plain P - KHP went
    /// indefinite on R1 wheel odometry and then stepped away from each reading.
    pub fn update_rows(
        &mut self,
        h: &CoreRows,
        residual: &DVector<f64>,
        variances: &DVector<f64>,
        scale: f64,
        gate: bool,
    ) -> Option<Innovation> {
        let rows = residual.len();
        if rows == 0 {
            return None;
        }
        let h_p = h * self.p;
        let predicted = &h_p * h.transpose();
        let spread = |scale: f64| {
            let mut s = predicted.clone();
            for i in 0..rows {
                s[(i, i)] += variances[i] * scale;
            }
            s.try_inverse()
        };
        let (Some(unscaled), Some(inverse)) = (spread(1.0), spread(scale)) else {
            return None;
        };
        let innovation = Innovation {
            squared: residual.dot(&(&unscaled * residual)),
            dof: rows,
            accepted: !gate
                || residual.dot(&(&inverse * residual)) <= CHI_SQUARED_99[rows.min(6) - 1],
        };
        if !innovation.accepted {
            return Some(innovation);
        }
        let (before, before_p) = (self.x.clone(), self.p);
        let gain = h_p.transpose() * &inverse;
        let step = &gain * residual;
        let keep = StateMatrix::identity() - &gain * h;
        self.p = keep * self.p * keep.transpose()
            + &gain * DMatrix::from_diagonal(&(variances * scale)) * gain.transpose();
        self.x.advance(&StateVector::from(step), 1.0);
        self.carry_extra(&before, &before_p);
        Some(innovation)
    }

    /// Rows observing the 30 states (`h`) and the extra ones (`h_extra`) together.
    fn update_joint(
        &mut self,
        h: &CoreRows,
        h_extra: &DMatrix<f64>,
        residual: &DVector<f64>,
        variances: &DVector<f64>,
    ) {
        let p_ht = self.p * h.transpose() + &self.p_extra * h_extra.transpose();
        let p_ht_extra =
            self.p_extra.transpose() * h.transpose() + &self.p_extra_extra * h_extra.transpose();
        let mut innovation = h * &p_ht + h_extra * &p_ht_extra;
        for i in 0..residual.len() {
            innovation[(i, i)] += variances[i];
        }
        let Some(inverse) = innovation.try_inverse() else {
            return;
        };
        let gain = &p_ht * &inverse;
        let gain_extra = &p_ht_extra * &inverse;
        let step = &gain * residual;
        self.x.advance(&StateVector::from(step), 1.0);
        self.extra += &gain_extra * residual;
        self.p -= &gain * p_ht.transpose();
        self.p_extra -= &gain * p_ht_extra.transpose();
        self.p_extra_extra -= &gain_extra * p_ht_extra.transpose();
    }

    /// Per-axis variances: zero leaves that axis unmeasured.
    fn measured_axes(variances: &[f64]) -> Vec<usize> {
        (0..variances.len())
            .filter(|&i| variances[i] > 0.0)
            .collect()
    }

    /// The body's angular velocity (rad/s, body frame), such as a camera's frame-to-frame
    /// rotation over its interval. False, and the state untouched, beyond the 99% bound of its
    /// predicted spread: a frozen or lost tracker, not a turn.
    pub fn update_angular_velocity(
        &mut self,
        rate: &Vector3<f64>,
        variances: &Vector3<f64>,
        scale: f64,
    ) -> Option<Innovation> {
        let axes = Self::measured_axes(variances.as_slice());
        let mut h = CoreRows::zeros(axes.len());
        let residual = DVector::from_iterator(
            axes.len(),
            axes.iter().map(|&a| rate[a] - self.x.angular_velocity[a]),
        );
        for (row, &a) in axes.iter().enumerate() {
            h[(row, 15 + a)] = 1.0;
        }
        let variances = DVector::from_iterator(axes.len(), axes.iter().map(|&a| variances[a]));
        self.update_rows(&h, &residual, &variances, scale, true)
    }

    /// The body's own velocity (m/s, body frame), such as wheeled odometry over its interval.
    /// Same 99% gate as a rate.
    pub fn update_body_velocity(
        &mut self,
        velocity: &Vector3<f64>,
        variances: &Vector3<f64>,
        scale: f64,
    ) -> Option<Innovation> {
        let body = self.x.rotation.transpose() * self.x.velocity;
        // Rotation on the right turns the measured vector the other way: d(R^T v)/dtheta = [R^T v]x.
        let turn = hat(&body);
        let to_body = self.x.rotation.transpose();
        let axes = Self::measured_axes(variances.as_slice());
        let mut h = CoreRows::zeros(axes.len());
        for (row, &a) in axes.iter().enumerate() {
            for c in 0..3 {
                h[(row, 3 + c)] = turn[(a, c)];
                h[(row, 12 + c)] = to_body[(a, c)];
            }
        }
        let residual =
            DVector::from_iterator(axes.len(), axes.iter().map(|&a| velocity[a] - body[a]));
        let variances = DVector::from_iterator(axes.len(), axes.iter().map(|&a| variances[a]));
        self.update_rows(&h, &residual, &variances, scale, true)
    }

    /// The body's pose in the filter's world: position along the world axes, rotation about
    /// the body's own [roll pitch yaw], with per-axis variances. Same 99% gate.
    pub fn update_pose(
        &mut self,
        position: &Vector3<f64>,
        rotation: &Matrix3<f64>,
        variances: &[f64; 6],
        scale: f64,
    ) -> Option<Innovation> {
        let turn = log(&(self.x.rotation.transpose() * rotation));
        let axes = Self::measured_axes(variances);
        let mut h = CoreRows::zeros(axes.len());
        let mut residual = DVector::zeros(axes.len());
        for (row, &a) in axes.iter().enumerate() {
            h[(row, a)] = 1.0;
            residual[row] = if a < 3 {
                position[a] - self.x.position[a]
            } else {
                turn[a - 3]
            };
        }
        let variances = DVector::from_iterator(axes.len(), axes.iter().map(|&a| variances[a]));
        self.update_rows(&h, &residual, &variances, scale, true)
    }
}

/// How a scan's points are matched to the map and weighed.
#[derive(Clone, Copy, Debug)]
pub struct LidarMeasurement {
    /// m: a fitted plane is used only when all its points lie this close to it.
    pub plane_tolerance: f32,
    /// A point is used only when its range exceeds this times the square of its distance to
    /// its plane: near points must lie closer to it than far ones.
    pub range_per_distance_sq: f64,
    /// m^2: variance of one point-to-plane distance.
    pub point_variance: f64,
    /// m: map points farther than this are not neighbors.
    pub neighbor_range: f64,
    pub mount_rotation: Matrix3<f64>,
    pub mount_translation: Vector3<f64>,
}

/// One group of a scan's points taken at the same instant, with the buffers that persist
/// across scans per point slot: its world position, its neighbors (left as they were when the
/// map has none near it) and whether it was used.
pub struct PointGroup<'a> {
    pub body: &'a [SurfacePoint],
    pub world: &'a mut [SurfacePoint],
    pub neighbors: &'a mut [Vec<SurfacePoint>],
    pub used: &'a mut [bool],
}

impl LioFilter {
    /// Matches `group` against `map` from the current pose and, when any point finds a plane,
    /// updates the state with the points' distances to their planes. True when it updated.
    pub fn update_points(
        &mut self,
        group: &mut PointGroup,
        map: &mut PointMap,
        m: &LidarMeasurement,
    ) -> bool {
        let mut planes = Vec::with_capacity(group.body.len());
        for j in 0..group.body.len() {
            let body = &group.body[j];
            let world = to_world(
                body,
                &self.x.rotation,
                &self.x.position,
                &m.mount_rotation,
                &m.mount_translation,
            );
            group.world[j].position = world.position;
            group.world[j].reflectivity = world.reflectivity;
            let lever = Vector3::new(
                body.position[0] as f64,
                body.position[1] as f64,
                body.position[2] as f64,
            );
            let range = norm(&lever);
            let near = &mut group.neighbors[j];
            map.nearest(&group.world[j], PLANE_POINTS, m.neighbor_range, near);
            group.used[j] = false;
            planes.push(None);
            if near.len() < PLANE_POINTS {
                continue;
            }
            if let Some(plane) = fit_plane(near, m.plane_tolerance) {
                let w = group.world[j].position;
                let distance =
                    (plane[0] * w[0] + plane[1] * w[1] + plane[2] * w[2] + plane[3]).abs();
                if range > m.range_per_distance_sq * distance as f64 * distance as f64 {
                    group.used[j] = true;
                    planes[j] = Some(plane);
                }
            }
        }
        let used = planes.iter().flatten().count();
        if used == 0 {
            return false;
        }
        // Rows over position and rotation (12 columns; mount columns stay zero).
        let mut h = DMatrix::<f64>::zeros(used, 12);
        let mut residual = DVector::<f64>::zeros(used);
        let mut row = 0;
        for (j, plane) in planes.iter().enumerate() {
            let Some(plane) = plane else {
                continue;
            };
            let normal = Vector3::new(plane[0] as f64, plane[1] as f64, plane[2] as f64);
            let body = &group.body[j];
            let lever = Vector3::new(
                body.position[0] as f64,
                body.position[1] as f64,
                body.position[2] as f64,
            );
            let in_body = m.mount_rotation * lever + m.mount_translation;
            let local_normal = self.x.rotation.transpose() * normal;
            let turn = hat(&in_body) * local_normal;
            let mut entries = SVector::<f64, 12>::zeros();
            entries.fixed_rows_mut::<3>(0).copy_from(&normal);
            entries.fixed_rows_mut::<3>(3).copy_from(&turn);
            h.row_mut(row).copy_from(&entries.transpose());
            let w = group.world[j].position;
            residual[row] = -normal[0] * w[0] as f64
                - normal[1] * w[1] as f64
                - normal[2] * w[2] as f64
                - plane[3] as f64;
            row += 1;
        }
        let noise = m.point_variance;
        let gain: OMatrix<f64, Const<STATE_DIM>, Dyn> = if STATE_DIM > used {
            let p_ht = self.p.fixed_view::<STATE_DIM, 12>(0, 0) * h.transpose();
            let mut innovation = &h * p_ht.rows(0, 12);
            for d in 0..used {
                innovation[(d, d)] += noise;
            }
            p_ht * innovation.try_inverse().expect("lidar innovation singular")
        } else {
            // Many points: through the information form, as Point-LIO weighs it.
            let information = noise * h.transpose() * &h;
            let mut inverse = self.p.try_inverse().expect("covariance singular");
            let mut pose_block = inverse.fixed_view_mut::<12, 12>(0, 0);
            pose_block += information.fixed_view::<12, 12>(0, 0);
            inverse = inverse.try_inverse().expect("information singular");
            inverse.fixed_view::<STATE_DIM, 12>(0, 0) * h.transpose() * noise
        };
        let step: OMatrix<f64, Const<STATE_DIM>, nalgebra::U1> = &gain * &residual;
        self.x.advance(&StateVector::from(step), 1.0);
        self.p -= gain * &h * self.p.fixed_view::<12, STATE_DIM>(0, 0);
        // The group's world positions from the updated pose.
        for j in 0..group.body.len() {
            let world = to_world(
                &group.body[j],
                &self.x.rotation,
                &self.x.position,
                &m.mount_rotation,
                &m.mount_translation,
            );
            group.world[j].position = world.position;
            group.world[j].reflectivity = world.reflectivity;
        }
        true
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn advancing_by_a_step_then_differencing_returns_the_step() {
        let start = LioState {
            rotation: exp(&Vector3::new(0.3, -0.2, 0.9)),
            mount_rotation: exp(&Vector3::new(0.01, 0.02, -0.03)),
            gravity: Vector3::new(0.0, 0.0, -9.81),
            ..LioState::default()
        };
        let step = StateVector::from_fn(|r, _| 0.01 * (r as f64 - 14.5));
        let mut moved = start.clone();
        moved.advance(&step, 1.0);
        assert!((moved.difference(&start) - step).norm() < 1e-12);
    }

    #[test]
    fn predicting_the_state_moves_it_along_its_rates() {
        let mut filter = LioFilter {
            x: LioState {
                velocity: Vector3::new(1.0, 0.0, 0.0),
                specific_force: Vector3::new(0.0, 0.0, 9.81),
                gravity: Vector3::new(0.0, 0.0, -9.81),
                ..LioState::default()
            },
            ..LioFilter::default()
        };
        let q = process_noise(20.0, 1000.0, 500.0, 0.0001, 0.0001);
        filter.predict(0.5, &q, true, true);
        assert!((filter.x.position - Vector3::new(0.5, 0.0, 0.0)).norm() < 1e-12);
        assert!(filter.p[(0, 0)] > 1.0);
    }
}
