//! Every scenario whose recording is on this machine must meet its expectations. Recordings
//! live outside the repo (`$DIMSLAM_DATASETS`, else `~/datasets`); a missing one is skipped,
//! and so is a slam scenario without `CUVSLAM_SDK_DIR`.

use dim_slam_replay::{
    datasets_dir, evaluate, ReplayOptions,
    scenario::{Mode, Scenario},
};

#[test]
fn every_present_scenario_meets_its_expectations() {
    let datasets = datasets_dir();
    let mut failures = Vec::new();
    let mut paths: Vec<_> = std::fs::read_dir(concat!(env!("CARGO_MANIFEST_DIR"), "/scenarios"))
        .expect("scenarios dir")
        .map(|entry| entry.expect("entry").path())
        .filter(|path| path.extension().is_some_and(|e| e == "json"))
        .collect();
    paths.sort();
    for path in paths {
        let scenario = Scenario::load(&path).expect("scenario parses");
        if !scenario.recording_path(&datasets).exists() {
            eprintln!("skip {}: no recording", scenario.name);
            continue;
        }
        if scenario.mode == Mode::Slam && std::env::var_os("CUVSLAM_SDK_DIR").is_none() {
            eprintln!("skip {}: CUVSLAM_SDK_DIR unset", scenario.name);
            continue;
        }
        let outcome = evaluate(&scenario, &[], &ReplayOptions::new(datasets.clone())).unwrap_or_else(|e| panic!("{e}"));
        failures.extend(
            outcome
                .failures
                .iter()
                .map(|f| format!("{}: {f}", scenario.name)),
        );
    }
    assert!(failures.is_empty(), "{failures:#?}");
}
