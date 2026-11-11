# Dataset and repository separation

The Hugging Face repository contains annotations and metadata only:

```
README.md, LICENSE, NOTICE.md, SHA256SUMS
categories.json, agent_instructions.txt, external.json
tasks/index.json, tasks/tasks.jsonl
tasks/<ID>/{prompt.txt,task.json,entities.json}
rubrics/<ID>/{task.json,gt.json,...}
```

The viewer configuration `tasks` exposes the complete test set. Load it with
`load_dataset("AidanYuzhe/vibeprevisbench", "tasks", split="test")`; download the full snapshot for evaluation.
`task_id` is stable. `kind` is text/image/continuation/video. `frames` is the generation frame count.
`prompt` and `rubric` paths in the index are relative to the dataset root; media paths in task.json are relative to its task directory.

Rubrics contain evaluator ground truth, never agent inputs. Media tasks include reference features under `refs/`;
script tasks have `claims.json`. The original rubric bytes and frozen photo brief remain the evaluator's authority.
The separate generation brief for photo tasks adds an orbit. Do not substitute it for the pinned evaluator brief.

The repository does not bundle third-party images/videos or executable download scripts. Fetch media using the GitHub
CLI. The manifest records source attribution, cuts and hashes. `SHA256SUMS` covers distributed dataset files, not
media fetched later. `verify` checks those bytes. See [external inputs](external-inputs.md).
