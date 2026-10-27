//! One self-contained interactive page per scenario: metrics table, x over time with the rest
//! stretches shaded, distance from the reference, and the paths.

use serde_json::json;

use crate::metrics::{aligned_reference, Metrics};
use crate::run::Run;

const COLUMNS: [&str; 9] = [
    "rest_p2p_mm_mean",
    "rest_p2p_deg_mean",
    "rest_step_rms_mm",
    "moving_roughness_rms_mm",
    "error_mm_mean",
    "error_mm_p95",
    "error_mm_max",
    "moving_path_m",
    "realtime_factor",
];
const COLORS: [&str; 6] = [
    "#2a7de1", "#e1572a", "#2aa876", "#9b59b6", "#c9a227", "#d6336c",
];

pub fn html(
    title: &str,
    runs: &[(Run, Metrics)],
    reference: &Metrics,
    segments: &[(i64, i64)],
    failures: &[String],
) -> String {
    let t0 = runs
        .first()
        .and_then(|(run, _)| run.fused.first())
        .map(|(t, _)| *t)
        .unwrap_or(0);
    let seconds = |t: i64| (t - t0) as f64 / 1e9;
    let (mut x_traces, mut error_traces, mut path_traces) = (Vec::new(), Vec::new(), Vec::new());
    if let Some((run, _)) = runs.first() {
        let aligned: Vec<_> = aligned_reference(run);
        let points: Vec<(f64, f64, f64)> = run
            .fused
            .iter()
            .zip(&aligned)
            .filter_map(|((t, _), r)| r.map(|r| (seconds(*t), r.translation.x, r.translation.y)))
            .collect();
        let style = json!({"color": "#888", "dash": "dot"});
        x_traces.push(json!({"name": "reference", "x": points.iter().map(|p| p.0).collect::<Vec<_>>(), "y": points.iter().map(|p| p.1 * 1000.0).collect::<Vec<_>>(), "line": style}));
        path_traces.push(json!({"name": "reference", "x": points.iter().map(|p| p.1).collect::<Vec<_>>(), "y": points.iter().map(|p| p.2).collect::<Vec<_>>(), "line": style}));
    }
    for ((run, _), color) in runs.iter().zip(COLORS.iter().cycle()) {
        let style = json!({"color": color});
        let ts: Vec<f64> = run.fused.iter().map(|(t, _)| seconds(*t)).collect();
        x_traces.push(json!({"name": run.variant, "x": ts, "y": run.fused.iter().map(|(_, p)| p.translation.x * 1000.0).collect::<Vec<_>>(), "line": style}));
        path_traces.push(json!({"name": run.variant, "x": run.fused.iter().map(|(_, p)| p.translation.x).collect::<Vec<_>>(), "y": run.fused.iter().map(|(_, p)| p.translation.y).collect::<Vec<_>>(), "line": style}));
        let errors: Vec<(f64, f64)> = run
            .fused
            .iter()
            .zip(aligned_reference(run))
            .filter_map(|((t, p), r)| {
                r.map(|r| {
                    (
                        seconds(*t),
                        (p.translation.vector.xy() - r.translation.vector.xy()).norm() * 1000.0,
                    )
                })
            })
            .collect();
        error_traces.push(json!({"name": run.variant, "x": errors.iter().map(|e| e.0).collect::<Vec<_>>(), "y": errors.iter().map(|e| e.1).collect::<Vec<_>>(), "line": style}));
    }
    let shapes: Vec<_> = segments.iter().map(|(a, b)| json!({"type": "rect", "xref": "x", "yref": "paper", "x0": seconds(*a), "x1": seconds(*b), "y0": 0, "y1": 1, "fillcolor": "#888", "opacity": 0.15, "line": {"width": 0}})).collect();
    let cell = |metrics: &Metrics, key: &str| {
        metrics
            .get(key)
            .filter(|v| v.is_finite())
            .map(|v| format!("{v:.2}"))
            .unwrap_or_default()
    };
    let mut rows = String::new();
    for (run, metrics) in runs {
        rows += &format!(
            "<tr><td>{}</td>{}</tr>",
            run.variant,
            COLUMNS
                .iter()
                .map(|k| format!("<td>{}</td>", cell(metrics, k)))
                .collect::<String>()
        );
    }
    rows += &format!(
        "<tr class=ref><td>reference</td>{}</tr>",
        COLUMNS
            .iter()
            .map(|k| format!("<td>{}</td>", cell(reference, k)))
            .collect::<String>()
    );
    let header: String = COLUMNS
        .iter()
        .map(|k| format!("<th>{}</th>", k.replace('_', " ")))
        .collect();
    let verdict = if failures.is_empty() {
        "<p class=pass>Every expectation met.</p>".to_string()
    } else {
        format!(
            "<p class=fail>Failed:</p><ul>{}</ul>",
            failures
                .iter()
                .map(|f| format!("<li>{f}</li>"))
                .collect::<String>()
        )
    };
    format!(
        r#"<!doctype html><html><head><meta charset=utf-8><meta name=viewport content="width=device-width,initial-scale=1"><title>{title}</title>
<script src="https://cdn.jsdelivr.net/npm/plotly.js-dist-min@2.35.2/plotly.min.js"></script>
<style>:root{{--bg:#fff;--fg:#222;--line:#ccc}} @media (prefers-color-scheme: dark){{:root{{--bg:#111;--fg:#ddd;--line:#444}}}}
body{{font-family:system-ui;margin:16px;background:var(--bg);color:var(--fg)}} .wrap{{overflow-x:auto}} table{{border-collapse:collapse;font-size:13px}} td,th{{border:1px solid var(--line);padding:3px 7px;text-align:right}} td:first-child{{text-align:left}} tr.ref{{opacity:.7}} .plot{{height:440px}} .pass{{color:#2aa876}} .fail{{color:#e1572a}}</style></head><body>
<h2>{title}</h2>{verdict}<p>Grey bands: the rest stream reads still. rest p2p = the output's extent within a band; moving roughness = rms second difference between publish ticks; error = distance from the reference aligned at the first sample.</p>
<div class=wrap><table><tr><th>variant</th>{header}</tr>{rows}</table></div>
<div id=x class=plot></div><div id=err class=plot></div><div id=path class=plot></div>
<script>
const dark = matchMedia('(prefers-color-scheme: dark)').matches;
const base = {{paper_bgcolor: dark ? '#111' : '#fff', plot_bgcolor: dark ? '#111' : '#fff', font: {{color: dark ? '#ddd' : '#222'}}, margin: {{l: 50, r: 10, t: 40, b: 40}}}};
Plotly.newPlot('x', {x}, {{...base, title: 'x (mm): zoom into a grey band to see the jitter', shapes: {shapes}, xaxis: {{title: 's'}}}}, {{responsive: true}});
Plotly.newPlot('err', {err}, {{...base, title: 'distance from the reference (mm)', shapes: {shapes}, xaxis: {{title: 's'}}}}, {{responsive: true}});
Plotly.newPlot('path', {path}, {{...base, title: 'path (m)', yaxis: {{scaleanchor: 'x'}}}}, {{responsive: true}});
</script></body></html>"#,
        x = serde_json::to_string(&x_traces).expect("json"),
        err = serde_json::to_string(&error_traces).expect("json"),
        path = serde_json::to_string(&path_traces).expect("json"),
        shapes = serde_json::to_string(&shapes).expect("json"),
    )
}
