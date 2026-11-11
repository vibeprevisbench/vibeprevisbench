# Evaluation protocol

## Rule-based Scene Check

Use Blender 5.2.2 to dump each delivered scene with `scorer/engine/vendor/v5_0_12/pvscore5/blender_dump.py`.
The public adapter calls the frozen pvscore 5.4 engine against the dataset's `rubrics/`. Rules, weights, thresholds
and the per-task members used by the paper are included in `scorer/engine/`.

Dimensions are A1 entities, A2 layout, A3 spaces, P1 picture structure, B1 routes, B2 events, C1 camera,
C2 cuts and D1 physical validity. Each task's Scene score is 100 times the mean of its applicable dimensions.
The benchmark mean includes all 242 tasks; absent and failed deliveries contribute zero.
`score --tasks ID1,ID2` produces a subset result, which must not be presented as the full benchmark result.

```bash
python vpbench.py score --data data --run runs/system --blender /path/to/blender --out results/system/scene
# Re-score existing scene dumps without Blender:
python vpbench.py score --data data --dumps dumps/system --out results/system/scene
```

Each dump directory contains `dump.json` and `dump.npz`. The adapter writes `scores.csv`, `summary.json` and the engine's
per-check reports. Read failed-task notes; configuration failures are not valid benchmark runs.

## VLM-based Screen Check

The two-turn judge first describes only the proxy video. It then receives the brief, reference media and checklists
for identity (V1), action (V2), visibility (V3) and story (V4). Photo tasks use V1 only and include the evaluator's orbit.
Checks already failed by the rule scorer are excluded. Yes/partial/no map to 1/0.5/0. A dimension averages its answered
entries; the task score averages the applicable dimensions and multiplies by 100. V2 excludes events whose V3 verdict
is no. The implementation preserves the frozen handling of n/a and missing entries.

The endpoint must implement Gemini generateContent. Configure `VPB_JUDGE_API_KEY` (or `GEMINI_API_KEY`),
`VPB_JUDGE_BASE_URL` (default Google v1beta), `VPB_JUDGE_MODEL` (paper: gemini-4-argon), and optionally
`VPB_JUDGE_AUTH=bearer`. Host ffmpeg and ffprobe must be on PATH. The judge makes paid calls to the selected provider.
Record endpoint, model, protocol and task selection when reporting a new result. Different models need not reproduce
the published judge scores.

`orbit` renders the deterministic two-part orbit of photo scenes before judging. `judge` reads the rule reports from
`--scores`; retain those reports. `report` uses all dataset tasks as the denominator and marks missing Screen
judgements as incomplete. Use a fresh output directory when code, dataset, submitted scenes or judge configuration change;
existing judge records and dumps are reused on restart.

## Photo protocol

The released generation briefs include an orbit. The frozen rubric brief is photo-only, matching the paper's agent
input. Both protocols use the same photo-frame scene scoring and evaluator-rendered orbit for Screen Check.
Do not count the agent's own orbit as evaluation evidence. Use `replay --protocol paper` for the original brief.
