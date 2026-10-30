// dim_slam's rpath does not reach a downstream binary, so point at the SDK's lib dir here.
fn main() {
    if let Ok(sdk) = std::env::var("CUVSLAM_SDK_DIR") {
        println!("cargo:rustc-link-arg=-Wl,-rpath,{sdk}/lib");
    }
    println!("cargo:rerun-if-env-changed=CUVSLAM_SDK_DIR");
}
