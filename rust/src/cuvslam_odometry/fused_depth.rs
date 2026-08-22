// Copyright 2026 Dimensional Inc.
// SPDX-License-Identifier: Apache-2.0
//
// RGB-guided densification of the raw depth image before the depth cloud is cut
// from it: the depth2depth crate predicts dense depth from the color image
// (Depth Anything V2), affine-anchors the prediction to the trusted raw pixels,
// keeps raw wherever the two agree and fills holes and outliers from the aligned
// prediction. Compiled to a stub unless the `depth2depth` cargo feature is on, so
// the model and its GPU backends stay out of builds that do not use them.

use lcm_msgs::sensor_msgs::Image;

pub struct Fuser {
    #[cfg(feature = "depth2depth")]
    model: depth2depth::Depth2Depth,
}

impl Fuser {
    /// None means disabled: a weight path is empty, the feature is compiled out, or
    /// the model failed to load (logged). The caller then publishes raw clouds.
    pub fn new(dinov2_weights: &str, head_weights: &str, quality: f64) -> Option<Self> {
        if dinov2_weights.is_empty() || head_weights.is_empty() {
            return None;
        }
        #[cfg(not(feature = "depth2depth"))]
        {
            let _ = quality;
            tracing::error!(
                "depth2depth weights are configured but dim_slam was built without the \
                 depth2depth feature; publishing raw depth clouds"
            );
            None
        }
        #[cfg(feature = "depth2depth")]
        {
            use depth2depth::candle::{utils, DType, Device};
            let device = if utils::cuda_is_available() {
                Device::new_cuda(0)
            } else if utils::metal_is_available() {
                Device::new_metal(0)
            } else {
                Ok(Device::Cpu)
            };
            let device = match device {
                Ok(device) => device,
                Err(error) => {
                    tracing::error!(%error, "depth2depth device init failed; publishing raw depth clouds");
                    return None;
                }
            };
            let dtype = if device.is_cpu() { DType::F32 } else { DType::F16 };
            let config = depth2depth::Config::default().with_quality(quality as f32);
            match depth2depth::Depth2Depth::new(dinov2_weights, head_weights, device, dtype, config)
            {
                Ok(model) => Some(Self { model }),
                Err(error) => {
                    tracing::error!(%error, "depth2depth weights failed to load; publishing raw depth clouds");
                    None
                }
            }
        }
    }

    /// The densified depth image (same resolution and units as the input), or None
    /// when the pieces are missing or inference fails — the caller falls back to the
    /// raw image, so a fault degrades quality rather than dropping the cloud.
    #[cfg(feature = "depth2depth")]
    pub fn fuse(&mut self, color: &Image, depth: &Image, units_per_meter: f64) -> Option<Image> {
        use dimos_module::warn_throttled;
        use std::time::Duration;

        let height = depth.height as usize;
        let width = depth.width as usize;
        let rgb = rgb_pixels(color, height, width)?;
        let step = depth.step as usize;
        let mut meters = vec![0f32; height * width];
        for v in 0..height {
            for u in 0..width {
                let at = v * step + u * 2;
                let raw = u16::from_ne_bytes([depth.data[at], depth.data[at + 1]]);
                meters[v * width + u] = raw as f32 / units_per_meter as f32;
            }
        }
        let fusion = match self.model.fuse(&rgb, &meters, height, width) {
            Ok(fusion) => fusion,
            Err(error) => {
                warn_throttled!(
                    Duration::from_secs(10),
                    error = %error,
                    "depth2depth inference failed; publishing this cloud from raw depth",
                );
                return None;
            }
        };
        let mut fused = depth.clone();
        fused.step = (width * 2) as i32;
        fused.data = Vec::with_capacity(height * width * 2);
        for z in &fusion.fused {
            let raw = (z * units_per_meter as f32).round().clamp(0.0, u16::MAX as f32) as u16;
            fused.data.extend_from_slice(&raw.to_ne_bytes());
        }
        Some(fused)
    }

    #[cfg(not(feature = "depth2depth"))]
    pub fn fuse(&mut self, _color: &Image, _depth: &Image, _units_per_meter: f64) -> Option<Image> {
        None
    }
}

/// Packed HxWx3 rgb bytes at the depth image's resolution, from an rgb8 or bgr8
/// color image.
#[cfg(feature = "depth2depth")]
fn rgb_pixels(color: &Image, height: usize, width: usize) -> Option<Vec<u8>> {
    use dimos_module::warn_throttled;
    use std::time::Duration;

    if color.height as usize != height || color.width as usize != width {
        warn_throttled!(
            Duration::from_secs(10),
            color = format!("{}x{}", color.width, color.height),
            depth = format!("{}x{}", width, height),
            "depth2depth needs color at the depth resolution; publishing raw depth clouds",
        );
        return None;
    }
    let step = color.step as usize;
    let mut rgb = vec![0u8; height * width * 3];
    match color.encoding.as_str() {
        "rgb8" => {
            for v in 0..height {
                rgb[v * width * 3..][..width * 3]
                    .copy_from_slice(&color.data[v * step..][..width * 3]);
            }
        }
        "bgr8" => {
            for v in 0..height {
                for u in 0..width {
                    let src = v * step + u * 3;
                    let dst = (v * width + u) * 3;
                    rgb[dst] = color.data[src + 2];
                    rgb[dst + 1] = color.data[src + 1];
                    rgb[dst + 2] = color.data[src];
                }
            }
        }
        other => {
            warn_throttled!(
                Duration::from_secs(10),
                encoding = %other,
                "depth2depth needs rgb8 or bgr8 color; publishing raw depth clouds",
            );
            return None;
        }
    }
    Some(rgb)
}
