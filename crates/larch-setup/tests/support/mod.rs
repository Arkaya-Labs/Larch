use larch_setup::manifest::Manifest;
use serde_json::json;
/// A small gated release shared by detection and planner fixtures.
pub fn manifest() -> Manifest {
    serde_json::from_value(json!({"schema":1,"min_setup":"0.0.0","version":"1.2.3","channel":"stable","components":{},"runtimes":{},
 "gpu_rules":[{"vendor":"nvidia","cc_min":"8.6","cc_max_excl":"9.0","driver_min":"580.65","flavour":"cuda","note":""},{"vendor":"nvidia","cc_min":"12.0","cc_max_excl":"13.0","driver_min":"580.65","flavour":"cuda","note":""},{"vendor":"nvidia","cc_min":"13.0","driver_min":"615.0","flavour":"cuda","note":"PTX JIT"}],"amd_ids":{"744C":"gfx1100"},
 "floors":{"macos_min":"26.0","glibc_min":"2.28","glibcxx_min":"3.4.29","windows_min_build":22000},
 "flavours":{"x86_64-unknown-linux-gnu":["cpu","cuda","hip"],"x86_64-pc-windows-msvc":["cpu","cuda","hip"],"aarch64-apple-darwin":["metal","cpu"]},"gated":["cpu","cuda","hip","metal"],"hip_targets":["gfx1030","gfx1100","gfx1101","gfx1102","gfx1200","gfx1201"],"models":[]})).unwrap()
}
