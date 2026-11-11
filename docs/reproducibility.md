# Reproducibility

The published leaderboard is a snapshot of the paper's results, not a new end-to-end run. The release keeps the rule
engine and its thresholds, the task-specific members, the pinned rubric briefs, the judge checklists and the judge
prompts used for the paper.

## What was checked before release

- **Rule-based Scene Check from the paper's scene dumps.** Rescoring all 242 GPT-6 Astra scenes with
  `scorer/score.py` gives the paper's Scene score (70.02; script 85.62, photos 55.75, continuation 79.76, video
  56.84). 238 of the 242 tasks match in every dimension. Four video tasks differ by at most 0.035 rule points; the
  frozen scoring package used for the paper gives the same values as this release on the same machine, so the
  difference comes from the environment of the paper's scoring run. All 98,957 per-check results recorded for the
  paper are identical.
- **Replay in Docker.** `harness/docker/Dockerfile` was built on a Docker host; its Blender binary is byte-identical
  to the one in the paper's sandbox. Four GPT-6 Astra deliveries, one per input kind, were prepared, collected,
  replayed and scored. The inputs prepared for the agent (photos, opening frame, video frame tiles) are
  byte-identical to the paper's runs. Three tasks score exactly as in the paper; the video task differs by 0.011
  rule points. Two replays of the same program on the same host differ by 0.004 points on that task: replayed meshes
  can differ in the last floating-point digits, and the area-weighted surface sampling of the scene dump turns this
  into small changes of continuous checks (routes, endpoints, framing distance).
- **VLM-based Screen Check.** Recomputing the scores from the paper's archived judge verdicts reproduces all 2,648
  judged task scores of the 11 agents. A new judge run calls the VLM API and is not deterministic.
- **scipy is required.** Without scipy the engine would fall back to greedy matching and a different convex hull,
  which changes scores, so `scorer/score.py` stops if scipy is missing. Scores were identical with numpy 2.4.4 +
  scipy 1.18.0 and numpy 2.5.4 + scipy 1.18.1.

## Comparing a new run

Record the dataset SHA256SUMS, code revision, Blender version, protocol (paper/release), task IDs, and the agent and
judge configurations. Keep all dump files and per-check engine reports. Compare per-task dimensions, not only
aggregate scores, and expect replay differences of the size above. External media reconstruction may produce
equivalent content with different encoded bytes; retain the download report and inspect mismatches. Media deleted or
restricted by a source cannot be guaranteed.

Published results count failed tasks as zero. Subset scores and incomplete Screen results must be labelled.
No alternative judge model or updated Blender version is asserted to reproduce the paper's numbers.
