// cargo:rustc-link-arg applies only to the package that emits it, so cu_vslam_rs's rpath
// never reaches our test binaries and they die at startup on @rpath/libcuvslam.dylib.
// A downstream binary has to emit this for itself.
use std::env;

fn main() {
    let lib_dir = env::var("DEP_CUVSLAM_LIB_DIR").expect("cu_vslam_rs exports lib_dir");
    println!("cargo:rustc-link-arg=-Wl,-rpath,{lib_dir}");
}
