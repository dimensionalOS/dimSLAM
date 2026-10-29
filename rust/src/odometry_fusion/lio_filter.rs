// Copyright 2026 Dimensional Inc.
// SPDX-License-Identifier: Apache-2.0
//
// Point-LIO's estimator, mirrored so the two can be compared stage by stage, bit for bit: the
// error-state filter over a 30-d state in which the IMU is a measurement of the body's angular
// velocity and specific force rather than its input. Each expression keeps Point-LIO's order
// of operations, since floating point is not associative; the quirks go once the outputs
// agree. Not to leave this branch while it mirrors (GPL upstream).

use nalgebra::{Matrix3, SMatrix, SVector, Vector3};

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

#[derive(Clone, Debug)]
pub struct LioFilter {
    pub x: LioState,
    pub p: StateMatrix,
}

impl Default for LioFilter {
    fn default() -> Self {
        Self {
            x: LioState::default(),
            p: StateMatrix::identity(),
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
        }
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
