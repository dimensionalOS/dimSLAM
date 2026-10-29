// Copyright 2026 Dimensional Inc.
// SPDX-License-Identifier: Apache-2.0

// The wheel carries libcuvslam and friends in dim_odom/_libs beside the extension
// module; cu_vslam_rs's own build.rs adds the build-time SDK path, which does not
// exist on an installed machine.
fn main() {
    let os = std::env::var("CARGO_CFG_TARGET_OS").unwrap_or_default();
    if os == "macos" {
        println!("cargo:rustc-link-arg=-Wl,-rpath,@loader_path/_libs");
    } else if os == "linux" {
        println!("cargo:rustc-link-arg=-Wl,-rpath,$ORIGIN/_libs");
    }
}
