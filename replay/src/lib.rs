// Copyright 2026 Dimensional Inc.
// SPDX-License-Identifier: Apache-2.0
//
// Replays dimos recordings through dimSLAM and scores each config variant against a reference.

pub mod db;
pub mod lcm;
pub mod metrics;
pub mod report;
pub mod run;
pub mod scenario;
pub mod tf;

use std::path::{Path, PathBuf};

use metrics::Metrics;
use run::Run;
use scenario::Scenario;

/// `$DIMSLAM_DATASETS`, else `~/datasets`.
pub fn datasets_dir() -> PathBuf {
    std::env::var_os("DIMSLAM_DATASETS")
        .map(PathBuf::from)
        .unwrap_or_else(|| {
            PathBuf::from(std::env::var_os("HOME").unwrap_or_default()).join("datasets")
        })
}

pub struct Outcome {
    pub runs: Vec<(Run, Metrics)>,
    pub reference: Metrics,
    pub segments: Vec<(i64, i64)>,
    pub failures: Vec<String>,
}

/// Every variant (or the named ones), scored and checked against the scenario's expectations.
pub fn evaluate(
    scenario: &Scenario,
    variants: &[String],
    datasets: &Path,
) -> Result<Outcome, String> {
    let names: Vec<String> = if variants.is_empty() {
        scenario.variants.keys().cloned().collect()
    } else {
        variants.to_vec()
    };
    let mut runs = Vec::new();
    let mut segments = Vec::new();
    for name in &names {
        let run = run::run(scenario, name, datasets)?;
        segments = metrics::rest_segments(&run, &scenario.rest);
        let scored = metrics::score(&run, &segments);
        runs.push((run, scored));
    }
    let reference = runs
        .first()
        .map(|(run, _)| metrics::score_reference(run, &segments))
        .unwrap_or_default();
    let mut failures = Vec::new();
    for (run, scored) in &runs {
        for (metric, bound) in scenario.expect.get(&run.variant).into_iter().flatten() {
            let Some(value) = scored.get(metric) else {
                failures.push(format!("{}: no metric {metric:?}", run.variant));
                continue;
            };
            // A NaN metric fails any bound.
            let outside = |limit: f64, beyond: std::cmp::Ordering| {
                value
                    .partial_cmp(&limit)
                    .is_none_or(|order| order == beyond)
            };
            if bound
                .max
                .is_some_and(|max| outside(max, std::cmp::Ordering::Greater))
                || bound
                    .min
                    .is_some_and(|min| outside(min, std::cmp::Ordering::Less))
            {
                failures.push(format!(
                    "{} {metric} = {value:.3}, expected {bound:?}",
                    run.variant
                ));
            }
        }
    }
    Ok(Outcome {
        runs,
        reference,
        segments,
        failures,
    })
}

/// metrics.json, one fused.jsonl per variant and report.html, under `out`.
pub fn write(scenario: &Scenario, outcome: &Outcome, out: &Path) -> std::io::Result<()> {
    use std::io::Write;
    std::fs::create_dir_all(out)?;
    let mut all = serde_json::Map::new();
    for (run, scored) in &outcome.runs {
        all.insert(
            run.variant.clone(),
            serde_json::to_value(scored).expect("json"),
        );
        let mut file = std::io::BufWriter::new(std::fs::File::create(
            out.join(format!("{}.jsonl", run.variant)),
        )?);
        for (t, pose) in &run.fused {
            let (p, q) = (pose.translation.vector, pose.rotation);
            writeln!(
                file,
                "{{\"t\":{t},\"pose\":[{},{},{},{},{},{},{}]}}",
                p.x, p.y, p.z, q.i, q.j, q.k, q.w
            )?;
        }
    }
    if let Some((run, _)) = outcome.runs.first() {
        let mut file = std::io::BufWriter::new(std::fs::File::create(out.join("reference.jsonl"))?);
        for (t, pose) in &run.reference {
            let (p, q) = (pose.translation.vector, pose.rotation);
            writeln!(
                file,
                "{{\"t\":{t},\"pose\":[{},{},{},{},{},{},{}]}}",
                p.x, p.y, p.z, q.i, q.j, q.k, q.w
            )?;
        }
    }
    all.insert(
        "reference".into(),
        serde_json::to_value(&outcome.reference).expect("json"),
    );
    all.insert(
        "failures".into(),
        serde_json::to_value(&outcome.failures).expect("json"),
    );
    std::fs::write(
        out.join("metrics.json"),
        serde_json::to_string_pretty(&all).expect("json"),
    )?;
    let title = format!("{}: {}", scenario.name, scenario.description);
    std::fs::write(
        out.join("report.html"),
        report::html(
            &title,
            &outcome.runs,
            &outcome.reference,
            &outcome.segments,
            &outcome.failures,
        ),
    )
}
