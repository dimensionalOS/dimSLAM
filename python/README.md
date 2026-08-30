# dim_odom

Python bindings for [dimSLAM](https://github.com/dimensionalOS/dimSLAM):
`CuvslamOdometry` wraps the cuVSLAM visual(-inertial) odometry front end,
`OdometryFusion` the error-state Kalman filter that fuses odometry sources with an
IMU. Everything runs in-process and on the caller's data stamps, so replaying a
recording is deterministic.

```python
import dim_odom

tracker = dim_odom.CuvslamOdometry(
    {"camera_mode": "stereo", "use_gpu": False},
    tf=lambda parent, child: ((0.0, 0.0, 0.0), (0.0, 0.0, 0.0, 1.0)),
)
tracker.handle_camera_info(dim_odom.CameraModel(...))
estimate = tracker.handle_image(dim_odom.ImageFrame(...))

fusion = dim_odom.OdometryFusion({"odom_sources": [{"parent_frame_id": "odom", "child_frame_id": "base_link"}]})
if estimate is not None:
    fusion.handle_source(estimate)
fused = fusion.maybe_publish()
```

Transforms cross the boundary as `((x, y, z), (qx, qy, qz, qw))`; the `tf` callable
answers rigid-mount lookups (`parent`, `child`) with such a tuple or `None`.

The wheels bundle `libcuvslam` ([open source](https://github.com/nvidia-isaac/cuVSLAM))
for each platform: macOS arm64 (CuMetal build; CPU tracking works with
`use_gpu: false`), manylinux x86_64 (CUDA 13; tracking needs a GPU), and manylinux
aarch64 (CUDA 13 Thor/JetPack 7 build). The Linux CUDA runtime comes from NVIDIA's
`nvidia-*` wheels, declared as dependencies, since those libraries exceed PyPI's
size limits. Configs are dicts mirroring the Rust `CuvslamOdometryConfig` and
`OdometryFusionConfig`; absent keys take the defaults.
