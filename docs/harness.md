# Agent workspace and replay

Build `harness/docker/Dockerfile` as `vibeprevisbench-sandbox:5.2.2`. Each task gets a fresh container. As in the
paper, the agent's container from `prepare` has 8 CPUs and 16 GB (`--cpus`, `--memory`; network `--network`, default
`bridge`), and each `replay` container has 4 CPUs and 8 GB with no network (`VPB_SANDBOX_CPUS`, `VPB_SANDBOX_MEMORY`,
`VPB_SANDBOX_NETWORK`). `VPB_SANDBOX_IMAGE` sets the image of both.
Blender is pinned to 5.2.2. The portable FFmpeg binary comes from imageio-ffmpeg 0.6.0.

`prepare` stages the brief, instructions, entity table, category table and fetched reference media. It exposes the
same managed Blender command and pvtools used for the paper. Start your chosen agent in the printed container with
`docker exec -it CONTAINER bash -l`. The task file is `/opt/task/TASK.md`, and the workspace is `/work/blender_agent`.
Do not mount the dataset's rubrics in the agent container. Agents write `solution/solution.py`.

`collect` copies the solution and workspace record into `runs/SYSTEM/ID` and removes the container unless `--keep`.
`replay` executes the submitted program in a fresh container and collects `scene.blend`, `video.mp4` and run records.
Missing solutions contribute zero in scoring. `--protocol paper` uses the paper brief; the default `release` uses the
released brief with the added photo orbit. `--force` replays completed deliveries.

For local inspection, `prepare` and `replay` accept `--backend local`, with Blender 5.2.2 and FFmpeg on PATH.
This executes submitted code on the host without isolation; use Docker for untrusted submissions.
The local backend is for debugging. The Docker image build and full isolated replay must be validated on your host
before claiming a paper-compatible run. `doctor` lists executable availability.

The original agent-side pvtools, runner, launcher, tool definitions and conventions are retained from the frozen runtime.
The published package replaces the original managed sandbox with Docker; this backend change requires environment
validation and does not by itself establish numerical equivalence.
