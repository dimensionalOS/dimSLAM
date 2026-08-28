// Copyright 2026 Dimensional Inc.
// SPDX-License-Identifier: Apache-2.0

//! Python bindings over the `dim_slam` crate: `CuvslamOdometry` wraps the cuVSLAM
//! front end, `OdometryFusion` the error-state Kalman fuser. Everything is
//! synchronous and driven by the caller's data stamps, so a replay is deterministic.
//!
//! Transforms cross the boundary as `((x, y, z), (qx, qy, qz, qw))`.

use dim_slam::nalgebra::{Isometry3, Matrix6, Quaternion, Translation3, UnitQuaternion, Vector3};
use dim_slam::types::{
    CameraModel as RsCameraModel, ImageFrame as RsImageFrame, ImuNoiseModel as RsImuNoiseModel,
    ImuSample as RsImuSample, OdometryEstimate as RsOdometryEstimate, PointCloud as RsPointCloud,
    Twist as RsTwist,
};
use dim_slam::{CuvslamCore, FusionCore};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyDict;

type PyTransform = ((f64, f64, f64), (f64, f64, f64, f64));

fn iso_from_py(((x, y, z), (qx, qy, qz, qw)): PyTransform) -> Isometry3<f64> {
    Isometry3::from_parts(
        Translation3::new(x, y, z),
        UnitQuaternion::from_quaternion(Quaternion::new(qw, qx, qy, qz)),
    )
}

fn iso_to_py(iso: &Isometry3<f64>) -> PyTransform {
    let t = iso.translation.vector;
    let q = iso.rotation.coords;
    ((t.x, t.y, t.z), (q.x, q.y, q.z, q.w))
}

fn matrix_to_row_major(m: &Matrix6<f64>) -> Vec<f64> {
    m.transpose().as_slice().to_vec()
}

fn matrix_from_row_major(v: Option<Vec<f64>>) -> PyResult<Matrix6<f64>> {
    match v {
        None => Ok(Matrix6::zeros()),
        Some(v) if v.len() == 36 => Ok(Matrix6::from_row_slice(&v)),
        Some(v) => Err(PyValueError::new_err(format!(
            "covariance needs 36 row-major values, got {}",
            v.len()
        ))),
    }
}

fn config_from_dict<T: serde::de::DeserializeOwned + Default>(
    config: Option<&Bound<'_, PyDict>>,
) -> PyResult<T> {
    match config {
        None => Ok(T::default()),
        Some(dict) => {
            pythonize::depythonize(dict).map_err(|e| PyValueError::new_err(format!("config: {e}")))
        }
    }
}

/// Calls back into a Python `tf(parent, child) -> ((x,y,z),(qx,qy,qz,qw)) | None`.
/// A Python exception inside the lookup answers "unconnected" rather than unwinding
/// through the Rust core mid-track; it is stored and re-raised after the handler.
struct PyTf<'py> {
    func: &'py Py<PyAny>,
    py: Python<'py>,
    error: std::cell::RefCell<Option<PyErr>>,
}

impl dim_slam::types::TfLookup for PyTf<'_> {
    fn latest(&self, parent: &str, child: &str) -> Option<Isometry3<f64>> {
        let result = self
            .func
            .call1(self.py, (parent, child))
            .and_then(|obj| obj.extract::<Option<PyTransform>>(self.py));
        match result {
            Ok(transform) => transform.map(iso_from_py),
            Err(err) => {
                self.error.borrow_mut().get_or_insert(err);
                None
            }
        }
    }
}

impl PyTf<'_> {
    fn rethrow(self) -> PyResult<()> {
        match self.error.into_inner() {
            Some(err) => Err(err),
            None => Ok(()),
        }
    }
}

/// A camera or depth image. `encoding` uses ROS names; anything but `mono8` is fed
/// to cuVSLAM as three-channel colour. `step` is bytes per row.
#[pyclass]
#[derive(Clone)]
struct ImageFrame {
    inner: RsImageFrame,
}

#[pymethods]
impl ImageFrame {
    #[new]
    #[pyo3(signature = (timestamp_ns, frame_id, width, height, encoding, step, data))]
    fn new(
        timestamp_ns: i64,
        frame_id: String,
        width: i32,
        height: i32,
        encoding: String,
        step: i32,
        data: Vec<u8>,
    ) -> Self {
        Self {
            inner: RsImageFrame {
                timestamp_ns,
                frame_id,
                width,
                height,
                encoding,
                step,
                data,
            },
        }
    }

    #[getter]
    fn timestamp_ns(&self) -> i64 {
        self.inner.timestamp_ns
    }

    #[getter]
    fn frame_id(&self) -> String {
        self.inner.frame_id.clone()
    }
}

/// Pinhole intrinsics plus plumb_bob distortion (k1, k2, p1, p2, k3).
/// `intrinsics` is the row-major 3x3: fx, 0, cx, 0, fy, cy, 0, 0, 1.
#[pyclass]
#[derive(Clone)]
struct CameraModel {
    inner: RsCameraModel,
}

#[pymethods]
impl CameraModel {
    #[new]
    #[pyo3(signature = (timestamp_ns, frame_id, width, height, distortion, intrinsics))]
    fn new(
        timestamp_ns: i64,
        frame_id: String,
        width: i32,
        height: i32,
        distortion: Vec<f64>,
        intrinsics: [f64; 9],
    ) -> Self {
        Self {
            inner: RsCameraModel {
                timestamp_ns,
                frame_id,
                width,
                height,
                distortion,
                intrinsics,
            },
        }
    }
}

#[pyclass]
#[derive(Clone)]
struct ImuSample {
    inner: RsImuSample,
}

#[pymethods]
impl ImuSample {
    #[new]
    #[pyo3(signature = (timestamp_ns, frame_id, angular_velocity, linear_acceleration))]
    fn new(
        timestamp_ns: i64,
        frame_id: String,
        angular_velocity: (f64, f64, f64),
        linear_acceleration: (f64, f64, f64),
    ) -> Self {
        Self {
            inner: RsImuSample {
                timestamp_ns,
                frame_id,
                angular_velocity: Vector3::new(
                    angular_velocity.0,
                    angular_velocity.1,
                    angular_velocity.2,
                ),
                linear_acceleration: Vector3::new(
                    linear_acceleration.0,
                    linear_acceleration.1,
                    linear_acceleration.2,
                ),
            },
        }
    }
}

/// cuVSLAM's inertial mode needs the sensor's noise model up front.
#[pyclass]
#[derive(Clone)]
struct ImuNoiseModel {
    inner: RsImuNoiseModel,
}

#[pymethods]
impl ImuNoiseModel {
    #[new]
    #[pyo3(signature = (frame_id, gyro_noise_density, gyro_random_walk, accel_noise_density, accel_random_walk, frequency))]
    fn new(
        frame_id: String,
        gyro_noise_density: f64,
        gyro_random_walk: f64,
        accel_noise_density: f64,
        accel_random_walk: f64,
        frequency: f64,
    ) -> Self {
        Self {
            inner: RsImuNoiseModel {
                frame_id,
                gyro_noise_density,
                gyro_random_walk,
                accel_noise_density,
                accel_random_walk,
                frequency,
            },
        }
    }
}

/// A pose of `child_frame_id` in `frame_id`, with the twist expressed in
/// `child_frame_id`. Covariances are 36 row-major values, xyz then rpy.
#[pyclass]
#[derive(Clone)]
struct OdometryEstimate {
    inner: RsOdometryEstimate,
}

#[pymethods]
impl OdometryEstimate {
    #[new]
    #[pyo3(signature = (timestamp_ns, frame_id, child_frame_id, translation, rotation_xyzw, pose_covariance=None, twist_linear=(0.0, 0.0, 0.0), twist_angular=(0.0, 0.0, 0.0), twist_covariance=None))]
    #[allow(clippy::too_many_arguments)]
    fn new(
        timestamp_ns: i64,
        frame_id: String,
        child_frame_id: String,
        translation: (f64, f64, f64),
        rotation_xyzw: (f64, f64, f64, f64),
        pose_covariance: Option<Vec<f64>>,
        twist_linear: (f64, f64, f64),
        twist_angular: (f64, f64, f64),
        twist_covariance: Option<Vec<f64>>,
    ) -> PyResult<Self> {
        Ok(Self {
            inner: RsOdometryEstimate {
                timestamp_ns,
                frame_id,
                child_frame_id,
                pose: iso_from_py((translation, rotation_xyzw)),
                pose_covariance: matrix_from_row_major(pose_covariance)?,
                twist: RsTwist {
                    linear: Vector3::new(twist_linear.0, twist_linear.1, twist_linear.2),
                    angular: Vector3::new(twist_angular.0, twist_angular.1, twist_angular.2),
                },
                twist_covariance: matrix_from_row_major(twist_covariance)?,
            },
        })
    }

    #[getter]
    fn timestamp_ns(&self) -> i64 {
        self.inner.timestamp_ns
    }

    #[getter]
    fn frame_id(&self) -> String {
        self.inner.frame_id.clone()
    }

    #[getter]
    fn child_frame_id(&self) -> String {
        self.inner.child_frame_id.clone()
    }

    #[getter]
    fn translation(&self) -> (f64, f64, f64) {
        let t = self.inner.pose.translation.vector;
        (t.x, t.y, t.z)
    }

    #[getter]
    fn rotation_xyzw(&self) -> (f64, f64, f64, f64) {
        let q = self.inner.pose.rotation.coords;
        (q.x, q.y, q.z, q.w)
    }

    #[getter]
    fn pose_covariance(&self) -> Vec<f64> {
        matrix_to_row_major(&self.inner.pose_covariance)
    }

    #[getter]
    fn twist_linear(&self) -> (f64, f64, f64) {
        let v = self.inner.twist.linear;
        (v.x, v.y, v.z)
    }

    #[getter]
    fn twist_angular(&self) -> (f64, f64, f64) {
        let v = self.inner.twist.angular;
        (v.x, v.y, v.z)
    }

    #[getter]
    fn twist_covariance(&self) -> Vec<f64> {
        matrix_to_row_major(&self.inner.twist_covariance)
    }

    fn __repr__(&self) -> String {
        let (x, y, z) = self.translation();
        format!(
            "OdometryEstimate({} -> {} @ {} ns, t=[{x:.4}, {y:.4}, {z:.4}])",
            self.inner.frame_id, self.inner.child_frame_id, self.inner.timestamp_ns
        )
    }
}

/// Range-gated depth points in the depth sensor's own frame.
#[pyclass]
struct PointCloud {
    inner: RsPointCloud,
}

#[pymethods]
impl PointCloud {
    #[getter]
    fn timestamp_ns(&self) -> i64 {
        self.inner.timestamp_ns
    }

    #[getter]
    fn frame_id(&self) -> String {
        self.inner.frame_id.clone()
    }

    #[getter]
    fn points(&self) -> Vec<(f32, f32, f32)> {
        self.inner
            .points
            .iter()
            .map(|p| (p[0], p[1], p[2]))
            .collect()
    }

    fn __len__(&self) -> usize {
        self.inner.points.len()
    }
}

/// The cuVSLAM front end. `config` mirrors the Rust `CuvslamOdometryConfig` as a
/// dict (absent keys default); `tf` answers rigid-mount lookups.
#[pyclass(unsendable)]
struct CuvslamOdometry {
    core: CuvslamCore,
    tf: Py<PyAny>,
}

impl CuvslamOdometry {
    fn with_tf<R>(
        &mut self,
        py: Python<'_>,
        run: impl FnOnce(&mut CuvslamCore, &PyTf<'_>) -> R,
    ) -> PyResult<R> {
        let tf = PyTf {
            func: &self.tf,
            py,
            error: std::cell::RefCell::new(None),
        };
        let out = run(&mut self.core, &tf);
        tf.rethrow()?;
        Ok(out)
    }
}

#[pymethods]
impl CuvslamOdometry {
    #[new]
    #[pyo3(signature = (config=None, *, tf))]
    fn new(config: Option<&Bound<'_, PyDict>>, tf: Py<PyAny>) -> PyResult<Self> {
        let config = config_from_dict(config)?;
        let core = CuvslamCore::new(config).map_err(PyValueError::new_err)?;
        Ok(Self { core, tf })
    }

    fn handle_camera_info(&mut self, py: Python<'_>, info: CameraModel) -> PyResult<()> {
        self.with_tf(py, |core, tf| core.handle_camera_info(info.inner, tf))
    }

    fn handle_imu_info(&mut self, py: Python<'_>, info: ImuNoiseModel) -> PyResult<()> {
        self.with_tf(py, |core, tf| core.handle_imu_info(info.inner, tf))
    }

    fn handle_imu(&mut self, imu: &ImuSample) {
        self.core.handle_imu(&imu.inner);
    }

    fn handle_image(
        &mut self,
        py: Python<'_>,
        image: ImageFrame,
    ) -> PyResult<Option<OdometryEstimate>> {
        let estimate = self.with_tf(py, |core, tf| core.handle_image(image.inner, tf))?;
        Ok(estimate.map(|inner| OdometryEstimate { inner }))
    }

    fn handle_depth_image(
        &mut self,
        py: Python<'_>,
        image: ImageFrame,
    ) -> PyResult<(Option<PointCloud>, Option<OdometryEstimate>)> {
        let (cloud, estimate) =
            self.with_tf(py, |core, tf| core.handle_depth_image(image.inner, tf))?;
        Ok((
            cloud.map(|inner| PointCloud { inner }),
            estimate.map(|inner| OdometryEstimate { inner }),
        ))
    }

    fn handle_depth_camera_info(&mut self, info: CameraModel) {
        self.core.handle_depth_camera_info(info.inner);
    }

    /// Logs counters (frames, tracked, drops) through the crate's tracing output.
    fn report(&self) {
        self.core.report();
    }
}

/// The error-state Kalman fuser. `config` mirrors the Rust `OdometryFusionConfig`
/// as a dict; `sources` keys are "parent->child" strings.
#[pyclass]
struct OdometryFusion {
    core: FusionCore,
}

#[pymethods]
impl OdometryFusion {
    #[new]
    #[pyo3(signature = (config=None))]
    fn new(config: Option<&Bound<'_, PyDict>>) -> PyResult<Self> {
        Ok(Self {
            core: FusionCore::new(config_from_dict(config)?),
        })
    }

    #[pyo3(signature = (imu, base_from_imu))]
    fn handle_imu(&mut self, imu: &ImuSample, base_from_imu: PyTransform) {
        self.core
            .handle_imu(&imu.inner, &iso_from_py(base_from_imu));
    }

    fn handle_source(&mut self, estimate: &OdometryEstimate) {
        self.core.handle_source(&estimate.inner);
    }

    /// The fused estimate once the publish period has elapsed on the data clock,
    /// else None. Call after each input.
    fn maybe_publish(&mut self) -> Option<OdometryEstimate> {
        self.core
            .maybe_publish()
            .map(|inner| OdometryEstimate { inner })
    }

    fn report(&self) {
        self.core.report();
    }
}

/// Sends the cores' tracing output to stderr. `level` is a tracing filter such as
/// "info", "debug", or "dim_slam=debug"; without this call the cores are silent.
#[pyfunction]
#[pyo3(signature = (level="info"))]
fn init_logging(level: &str) -> PyResult<()> {
    let filter = tracing_subscriber::EnvFilter::try_new(level)
        .map_err(|e| PyValueError::new_err(format!("level: {e}")))?;
    tracing_subscriber::fmt()
        .with_env_filter(filter)
        .with_writer(std::io::stderr)
        .try_init()
        .map_err(|e| PyValueError::new_err(e.to_string()))
}

/// Compose two transforms.
#[pyfunction]
fn compose(a: PyTransform, b: PyTransform) -> PyTransform {
    iso_to_py(&(iso_from_py(a) * iso_from_py(b)))
}

/// Invert a transform.
#[pyfunction]
fn invert(t: PyTransform) -> PyTransform {
    iso_to_py(&iso_from_py(t).inverse())
}

#[pymodule]
#[pyo3(name = "_native")]
fn dim_odom(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;
    m.add_class::<ImageFrame>()?;
    m.add_class::<CameraModel>()?;
    m.add_class::<ImuSample>()?;
    m.add_class::<ImuNoiseModel>()?;
    m.add_class::<OdometryEstimate>()?;
    m.add_class::<PointCloud>()?;
    m.add_class::<CuvslamOdometry>()?;
    m.add_class::<OdometryFusion>()?;
    m.add_function(wrap_pyfunction!(init_logging, m)?)?;
    m.add_function(wrap_pyfunction!(compose, m)?)?;
    m.add_function(wrap_pyfunction!(invert, m)?)?;
    Ok(())
}
