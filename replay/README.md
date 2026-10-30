# dim_slam_replay

Replays dimos recordings (memory2 `.db`) through dimSLAM and scores each config variant. LCM is
decoded by hand (`src/lcm.rs`) and nothing waits on the wall clock, so a run goes as fast as
the core does: fusion-only scenarios replay hundreds of times faster than real time.

```sh
cargo run --release -- scenarios/*.json            # every variant of every scenario
cargo run --release -- scenarios/r1_nav_20261015.json --variant imm --out out
cargo test --release                               # every scenario whose recording is here
```

Recordings stay out of git. They are found under `$DIMSLAM_DATASETS` (default `~/datasets`), and
a scenario whose recording is missing is skipped. `slam` scenarios run cuVSLAM, so they need
`CUVSLAM_SDK_DIR` (and on macOS `CUMETAL_PREBUILT_CACHE_DIR=$CUVSLAM_SDK_DIR/share/cumetal-cache`)
set at build and run time; without them the test skips them.

Each run writes `out/<scenario>/report.html` (interactive), `metrics.json` and one
`<variant>.jsonl` of output poses.

## Scenarios

A scenario (`scenarios/*.json`) names the recording, which streams feed the core and how
(`frame_id` overrides, `carry_to` a frame through the recording's tf, a camera_info's
`distortion_model` when the recording mislabels the lens), the reference the output is scored
against (never fed in), what counts as rest, the core's config, `variants` as JSON-pointer edits
of that config (null removes a key), and `expect`: per variant, bounds on metrics. `baseline` records what an earlier build scored, which the bounds were set from.

Metrics:
- `rest_p2p_mm_mean` / `_max`, `rest_p2p_deg_mean`: the output's extent within each stretch the
  rest stream reads still. Jitter.
- `rest_step_rms_mm`: tick-to-tick motion while at rest.
- `moving_roughness_rms_mm`: rms second difference between publish ticks while moving.
- `error_mm_*`: horizontal distance from the reference, aligned at the first output.
- `ate_xy_mm_*`, `ate_z_mm_mean`: the same under the yaw and offset that fit the whole run best,
  so they do not depend on how the two frames were oriented at the start.
- `realtime_factor`: recording seconds per wall second.

A baseline from another build: copy this directory into a checkout of that build and run it
there; its config must parse under that build, so edit the variants to match.
