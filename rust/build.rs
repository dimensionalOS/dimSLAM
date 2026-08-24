// cu_vslam_rs embeds the SDK rpath with cargo:rustc-link-arg, which cargo applies only
// to its own package. Without repeating it here, dim_slam links but dies at startup on
// @rpath/libcuvslam.dylib. cu_vslam_rs declares links = "cuvslam" and exports the
// directory, so the path arrives as DEP_CUVSLAM_LIB_DIR and is never named twice.
use std::env;

fn main() {
    let lib_dir = env::var("DEP_CUVSLAM_LIB_DIR").expect("cu_vslam_rs exports lib_dir");
    println!("cargo:rustc-link-arg=-Wl,-rpath,{lib_dir}");
}
