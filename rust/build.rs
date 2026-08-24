// cu_vslam_rs embeds the SDK rpath with cargo:rustc-link-arg, which cargo applies only
// to its own package. Without repeating it here, dim_slam links but dies at startup on
// @rpath/libcuvslam.dylib.
use std::env;
use std::path::PathBuf;

fn main() {
    println!("cargo:rerun-if-env-changed=CUVSLAM_SDK_DIR");
    let Ok(sdk_dir) = env::var("CUVSLAM_SDK_DIR") else {
        return;
    };
    let lib_dir = PathBuf::from(sdk_dir).join("lib");
    println!("cargo:rustc-link-arg=-Wl,-rpath,{}", lib_dir.display());
}
