//! Hand decoders for the LCM messages a replay needs. Big-endian, fields in declaration order,
//! the top-level message prefixed by its 8-byte fingerprint; strings are a length (with the
//! trailing NUL) and the bytes.

use dim_slam::nalgebra::{Isometry3, Matrix6, Quaternion, Translation3, UnitQuaternion, Vector3};
use dim_slam::{CameraModel, ImageFrame, ImuSample, LidarScan, OdometryEstimate, Twist};

pub type Result<T> = std::result::Result<T, String>;

struct Reader<'a> {
    buf: &'a [u8],
    at: usize,
}

impl<'a> Reader<'a> {
    fn message(buf: &'a [u8]) -> Self {
        Self { buf, at: 8 }
    }

    fn take(&mut self, n: usize) -> Result<&'a [u8]> {
        let end = self.at + n;
        let bytes = self.buf.get(self.at..end).ok_or("message truncated")?;
        self.at = end;
        Ok(bytes)
    }

    fn i32(&mut self) -> Result<i32> {
        Ok(i32::from_be_bytes(
            self.take(4)?.try_into().expect("4 bytes"),
        ))
    }

    fn u8(&mut self) -> Result<u8> {
        Ok(self.take(1)?[0])
    }

    fn f64(&mut self) -> Result<f64> {
        Ok(f64::from_be_bytes(
            self.take(8)?.try_into().expect("8 bytes"),
        ))
    }

    fn f64s<const N: usize>(&mut self) -> Result<[f64; N]> {
        let mut out = [0.0; N];
        for value in &mut out {
            *value = self.f64()?;
        }
        Ok(out)
    }

    fn string(&mut self) -> Result<String> {
        let len = self.i32()?;
        let bytes = self.take(usize::try_from(len).map_err(|_| "negative string length")?)?;
        let text = bytes.strip_suffix(&[0]).unwrap_or(bytes);
        Ok(String::from_utf8_lossy(text).into_owned())
    }

    /// std_msgs.Header: seq, stamp (sec, nsec), frame_id. Returns (stamp ns, frame_id).
    fn header(&mut self) -> Result<(i64, String)> {
        let _seq = self.i32()?;
        let sec = self.i32()?;
        let nsec = self.i32()?;
        Ok((
            i64::from(sec) * 1_000_000_000 + i64::from(nsec),
            self.string()?,
        ))
    }

    fn vector3(&mut self) -> Result<Vector3<f64>> {
        Ok(Vector3::from(self.f64s::<3>()?))
    }

    fn quaternion(&mut self) -> Result<UnitQuaternion<f64>> {
        let [x, y, z, w] = self.f64s::<4>()?;
        Ok(UnitQuaternion::from_quaternion(Quaternion::new(w, x, y, z)))
    }

    fn pose(&mut self) -> Result<Isometry3<f64>> {
        let position = self.vector3()?;
        Ok(Isometry3::from_parts(
            Translation3::from(position),
            self.quaternion()?,
        ))
    }

    fn covariance(&mut self) -> Result<Matrix6<f64>> {
        Ok(Matrix6::from_row_slice(&self.f64s::<36>()?))
    }
}

/// nav_msgs.Odometry.
pub fn odometry(buf: &[u8]) -> Result<OdometryEstimate> {
    let mut r = Reader::message(buf);
    let (timestamp_ns, frame_id) = r.header()?;
    let child_frame_id = r.string()?;
    let pose = r.pose()?;
    let pose_covariance = r.covariance()?;
    let linear = r.vector3()?;
    let angular = r.vector3()?;
    let twist_covariance = r.covariance()?;
    Ok(OdometryEstimate {
        timestamp_ns,
        frame_id,
        child_frame_id,
        pose,
        pose_covariance,
        twist: Twist { linear, angular },
        twist_covariance,
    })
}

/// sensor_msgs.Imu.
pub fn imu(buf: &[u8]) -> Result<ImuSample> {
    let mut r = Reader::message(buf);
    let (timestamp_ns, frame_id) = r.header()?;
    r.take(8 * (4 + 9))?;
    let angular_velocity = r.vector3()?;
    r.take(8 * 9)?;
    let linear_acceleration = r.vector3()?;
    Ok(ImuSample {
        timestamp_ns,
        frame_id,
        angular_velocity,
        linear_acceleration,
    })
}

pub struct Transform {
    pub timestamp_ns: i64,
    pub parent: String,
    pub child: String,
    /// parent_from_child.
    pub transform: Isometry3<f64>,
}

/// tf2_msgs.TFMessage.
pub fn tf(buf: &[u8]) -> Result<Vec<Transform>> {
    let mut r = Reader::message(buf);
    let count = r.i32()?;
    (0..count)
        .map(|_| {
            let (timestamp_ns, parent) = r.header()?;
            let child = r.string()?;
            let transform = r.pose()?;
            Ok(Transform {
                timestamp_ns,
                parent,
                child,
                transform,
            })
        })
        .collect()
}

/// sensor_msgs.Image.
pub fn image(buf: &[u8]) -> Result<ImageFrame> {
    let mut r = Reader::message(buf);
    let data_length = r.i32()?;
    let (timestamp_ns, frame_id) = r.header()?;
    let height = r.i32()?;
    let width = r.i32()?;
    let encoding = r.string()?;
    let _is_bigendian = r.u8()?;
    let step = r.i32()?;
    let data = r
        .take(usize::try_from(data_length).map_err(|_| "negative image length")?)?
        .to_vec();
    if encoding == "jpeg" {
        return jpeg_frame(timestamp_ns, frame_id, &data);
    }
    Ok(ImageFrame {
        timestamp_ns,
        frame_id,
        width,
        height,
        encoding,
        step,
        data,
    })
}

/// A JPEG payload as rgb8, or mono8 for a grayscale one.
fn jpeg_frame(timestamp_ns: i64, frame_id: String, jpeg: &[u8]) -> Result<ImageFrame> {
    let mut decoder = jpeg_decoder::Decoder::new(jpeg);
    let pixels = decoder.decode().map_err(|e| format!("jpeg: {e}"))?;
    let info = decoder.info().ok_or("jpeg: no header")?;
    let (encoding, channels) = match info.pixel_format {
        jpeg_decoder::PixelFormat::L8 => ("mono8", 1),
        jpeg_decoder::PixelFormat::RGB24 => ("rgb8", 3),
        other => return Err(format!("jpeg: unsupported pixel format {other:?}")),
    };
    Ok(ImageFrame {
        timestamp_ns,
        frame_id,
        width: i32::from(info.width),
        height: i32::from(info.height),
        encoding: encoding.to_string(),
        step: i32::from(info.width) * channels,
        data: pixels,
    })
}

/// sensor_msgs.CameraInfo, keeping what dimSLAM reads: size, distortion model and coefficients, and K.
pub fn camera_info(buf: &[u8]) -> Result<CameraModel> {
    let mut r = Reader::message(buf);
    let distortion_length = r.i32()?;
    let (timestamp_ns, frame_id) = r.header()?;
    let height = r.i32()?;
    let width = r.i32()?;
    let distortion_model = r.string()?;
    let distortion = (0..distortion_length)
        .map(|_| r.f64())
        .collect::<Result<Vec<_>>>()?;
    let intrinsics = r.f64s::<9>()?;
    Ok(CameraModel {
        timestamp_ns,
        frame_id,
        width,
        height,
        distortion_model,
        distortion,
        intrinsics,
    })
}

/// sensor_msgs.PointCloud2 as a lidar scan: float32 x/y/z and, when present, a uint32
/// `offset_time` (ns after the stamp, as Livox drivers write it). Little-endian data only.
pub fn lidar_scan(buf: &[u8]) -> Result<LidarScan> {
    let mut r = Reader::message(buf);
    let fields_length = r.i32()?;
    let data_length = r.i32()?;
    let (timestamp_ns, frame_id) = r.header()?;
    let height = r.i32()?;
    let width = r.i32()?;
    let mut offsets = std::collections::HashMap::new();
    for _ in 0..fields_length {
        let name = r.string()?;
        let offset = r.i32()?;
        let datatype = r.u8()?;
        let _count = r.i32()?;
        offsets.insert(name, (offset as usize, datatype));
    }
    if r.u8()? != 0 {
        return Err("big-endian point cloud".into());
    }
    let point_step = r.i32()? as usize;
    let _row_step = r.i32()?;
    let data = r.take(usize::try_from(data_length).map_err(|_| "negative cloud length")?)?;
    const FLOAT32: u8 = 7;
    const UINT32: u8 = 6;
    let field = |name: &str, datatype: u8| -> Result<usize> {
        match offsets.get(name) {
            Some((offset, kind)) if *kind == datatype => Ok(*offset),
            _ => Err(format!("cloud has no {name} field of type {datatype}")),
        }
    };
    let (x, y, z) = (field("x", FLOAT32)?, field("y", FLOAT32)?, field("z", FLOAT32)?);
    let time = field("offset_time", UINT32).ok();
    const UINT8: u8 = 2;
    let tag = field("tag", UINT8).ok();
    let intensity = field("intensity", FLOAT32).ok();
    let count = (width.max(0) as usize) * (height.max(0) as usize);
    if point_step == 0 || data.len() < count * point_step {
        return Err(format!("cloud data has {} bytes for {count} points", data.len()));
    }
    let f32_at = |p: &[u8], at: usize| f32::from_le_bytes(p[at..at + 4].try_into().expect("4"));
    let mut points = Vec::with_capacity(count);
    let mut offsets_ns = Vec::new();
    let mut tags = Vec::new();
    let mut intensities = Vec::new();
    for p in data.chunks_exact(point_step).take(count) {
        if let Some(at) = intensity {
            intensities.push(f32_at(p, at));
        }
        if let Some(at) = tag {
            tags.push(p[at]);
        }
        points.push([f32_at(p, x), f32_at(p, y), f32_at(p, z)]);
        if let Some(at) = time {
            offsets_ns.push(u32::from_le_bytes(p[at..at + 4].try_into().expect("4")));
        }
    }
    Ok(LidarScan {
        timestamp_ns,
        frame_id,
        points,
        offsets_ns,
        tags,
        intensities,
    })
}

/// A PointCloud2's header stamp, without decoding its points.
pub fn lidar_stamp(buf: &[u8]) -> Result<i64> {
    let mut r = Reader::message(buf);
    r.take(8)?;
    Ok(r.header()?.0)
}

/// geometry_msgs.PoseStamped: (stamp ns, pose).
pub fn pose_stamped(buf: &[u8]) -> Result<(i64, Isometry3<f64>)> {
    let mut r = Reader::message(buf);
    let (timestamp_ns, _frame_id) = r.header()?;
    Ok((timestamp_ns, r.pose()?))
}
