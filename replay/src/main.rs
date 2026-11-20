//! dim_slam_replay <scenario.json>... [--variant NAME]... [--out DIR] [--datasets DIR]
//!   [--perturb-seed N [--perturb-scan I]] [--turns-record DIR] [--turns-play DIR]
//! Replays each scenario's recording through every variant (or the named ones), prints the
//! metrics, writes report.html + metrics.json + each variant's output under DIR/<scenario>
//! (default `out/`), and exits 1 if any expectation fails. --perturb-seed drops one point
//! (seed-picked) of scan I (default 8); --turns-record/--turns-play cache the camera tracker.

use std::path::PathBuf;

use dim_slam_replay::{datasets_dir, evaluate, scenario::Scenario, write, ReplayOptions};

fn main() {
    tracing_subscriber::fmt()
        .with_writer(std::io::stderr)
        .with_max_level(tracing::Level::INFO)
        .init();
    let mut args = std::env::args().skip(1);
    let (mut scenarios, mut variants) = (Vec::new(), Vec::new());
    let mut out = PathBuf::from("out");
    let mut options = ReplayOptions::new(datasets_dir());
    let (mut perturb_seed, mut perturb_scan) = (None, 8u64);
    while let Some(arg) = args.next() {
        let mut value = || args.next().unwrap_or_else(|| panic!("{arg} needs a value"));
        match arg.as_str() {
            "--variant" => variants.push(value()),
            "--out" => out = PathBuf::from(value()),
            "--datasets" => options.datasets = PathBuf::from(value()),
            "--perturb-seed" => perturb_seed = Some(value().parse().expect("--perturb-seed: a number")),
            "--perturb-scan" => perturb_scan = value().parse().expect("--perturb-scan: a number"),
            "--turns-record" => options.turns_record = Some(PathBuf::from(value())),
            "--turns-play" => options.turns_play = Some(PathBuf::from(value())),
            "-h" | "--help" => {
                println!(
                    "{}",
                    include_str!("main.rs")
                        .lines()
                        .take(7)
                        .map(|l| l.trim_start_matches("//!").trim())
                        .collect::<Vec<_>>()
                        .join("\n")
                );
                return;
            }
            _ => scenarios.push(PathBuf::from(arg)),
        }
    }
    options.perturb = perturb_seed.map(|seed| (perturb_scan, seed));
    let mut failed = false;
    for path in &scenarios {
        let scenario = Scenario::load(path).unwrap_or_else(|e| panic!("{e}"));
        let recording = scenario.recording_path(&options.datasets);
        if !recording.exists() {
            println!(
                "{}: skipped, no recording at {}",
                scenario.name,
                recording.display()
            );
            continue;
        }
        let outcome = evaluate(&scenario, &variants, &options)
            .unwrap_or_else(|e| panic!("{}: {e}", scenario.name));
        let dir = out.join(&scenario.name);
        write(&scenario, &outcome, &dir).unwrap_or_else(|e| panic!("{}: {e}", dir.display()));
        println!(
            "\n{} ({})",
            scenario.name,
            dir.join("report.html").display()
        );
        let keys: Vec<&String> = outcome
            .reference
            .keys()
            .chain(
                outcome
                    .runs
                    .first()
                    .map(|(_, m)| m.keys())
                    .into_iter()
                    .flatten(),
            )
            .collect::<std::collections::BTreeSet<_>>()
            .into_iter()
            .collect();
        print!("{:<26}", "metric");
        for (run, _) in &outcome.runs {
            print!("{:>14}", run.variant);
        }
        println!("{:>14}", "reference");
        for key in keys {
            print!("{key:<26}");
            for (_, metrics) in &outcome.runs {
                print!(
                    "{:>14}",
                    metrics
                        .get(key)
                        .map(|v| format!("{v:.2}"))
                        .unwrap_or_default()
                );
            }
            println!(
                "{:>14}",
                outcome
                    .reference
                    .get(key)
                    .map(|v| format!("{v:.2}"))
                    .unwrap_or_default()
            );
        }
        for (run, _) in &outcome.runs {
            for why in run.dropped.iter().take(3) {
                println!("  {} dropped: {why}", run.variant);
            }
        }
        for failure in &outcome.failures {
            println!("  FAIL {failure}");
        }
        failed |= !outcome.failures.is_empty();
    }
    std::process::exit(i32::from(failed));
}
