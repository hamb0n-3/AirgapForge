---
name: webdev-pipeline
description: "Train web-dev knowledge into a local model on Apple Silicon: build a web-dev instruction dataset, train LoRA adapters with MLX, and score them with a local eval harness. Runs on the shared pipeline engine from this domain's manifest, so a stage is one command and the engine is the same one the other domains use."
version: 0.1.0
author: Tristan
license: MIT
platforms: [macos]
metadata:
  tags: [Training, MLX, LoRA, Dataset, Evaluation, WebDev]
  related_skills: [airgapforge-cyber]
---

# Web-dev pipeline

This is a starting template. It is the web-dev instance of the reusable pipeline: the domain-agnostic engine lives in `_pipeline-engine/`, and everything specific to web-dev is described in `manifest.yaml`. The engine gates on preflight, sets `WANDB_MODE=offline`, skips finished steps, runs the model matrix, and backgrounds long stages to a log. You do not edit the engine; you fill in the manifest and write the scripts it names.

It has three stages, not four: `data`, `train`, `eval`. There is no offensive stage, because that is a cyber-only concept. Compared with the cyber pipeline, the trainer and the data-plumbing steps carry over, data collection and eval are new, and the model matrix is simpler.

## What to reuse and what to write

The manifest tags each step, and the split is the point of the template:

- **Reuse** (copy the generic scripts unchanged): the review gate, dataset assembly, chat-template conversion, and the LoRA trainer. None of these care about the domain.
- **Adapt** (same code, new prompts): the two-pass filter and the instruction/response structurer. The shape is generic; the prompts are web-dev prompts.
- **Write** (new for this domain): the collector (web-dev sources like MDN, framework docs, Stack Overflow, GitHub, RFCs are not cyber sources), and the eval benchmarks (a coding or knowledge set). The generic LLM-judge quality metric can carry over.

## Paths

The engine, this manifest, and the web-dev code repo are three separate things. The first two get migrated into `~/dev/sora/skills/`; the code repo is named by the manifest's `repo_root` (override with `WEBDEV_ROOT`), so it can live anywhere. Set three shell variables:

```sh
PY=~/.venv/bin/python                                # has mlx and pyyaml; runs the engine
ENG=~/dev/sora/skills/_pipeline-engine/pipeline.py
MAN=~/dev/sora/skills/webdev-pipeline/manifest.yaml
export WEBDEV_ROOT=/path/to/webdev-forge             # the repo holding the scripts the manifest names
```

## Setup (once)

Apple Silicon with enough memory for the bases you pick (web-dev is lighter than the cyber matrix; a 4B fits in 8 GB, an 8B in 16 GB). Then a virtualenv:

```sh
python3 -m venv ~/.venv
~/.venv/bin/pip install -r "$WEBDEV_ROOT"/requirements.txt   # needs mlx, mlx-lm, pyyaml at least
```

## Run it

Preflight first; its `requires` failures are your to-do list of scripts still to write. Then a stage at a time:

```sh
$PY ~/dev/sora/skills/_pipeline-engine/preflight.py --manifest $MAN --stages all --tier all

$PY $ENG --manifest $MAN --stage data                 # collect, filter, structure, assemble, templates (no review)
$PY $ENG --manifest $MAN --stage data --pause         # stop after structure for the human review
$PY $ENG --manifest $MAN --stage train                # train every base in the matrix; --only <key> for one
$PY $ENG --manifest $MAN --stage eval -- --limit 200  # flags after -- pass straight to your eval runner
```

Each stage backgrounds itself and returns a log path under `~/.pipeline/webdev-pipeline/logs/`, so `tail -f` it to watch. Useful flags: `--dry-run` prints the exact commands without running them, `--foreground` runs inline, `--force` re-runs a step whose output exists, `--only <key>` limits training to one base, `--pause` stops at the review gate, and `--tier 8B` sizes the check to one tier.

## Filling it in

1. Point `repo_root` in `manifest.yaml` (or `WEBDEV_ROOT`) at your web-dev repo.
2. Write the `NEW` and `ADAPT` scripts at the names the manifest lists, or rename the manifest entries to match the scripts you already have. The manifest just names commands; the engine runs whatever you point it at.
3. Give the `train` stage a `config.py` whose `MODELS` list emits one row per base (the manifest's `list_cmd` reads `key` and `repo`).
4. Re-run preflight until it is READY, then run the stages.

For the full field reference (steps, gates, matrix, passthrough), see `_pipeline-engine/schema.md`.

## Migrating into sora

Move the folders into sora's skills directory (the web-dev code repo stays put):

```sh
mv "$AIRGAPFORGE_ROOT/skills/_pipeline-engine"  ~/dev/sora/skills/   # if not already moved
mv "$AIRGAPFORGE_ROOT/skills/webdev-pipeline"   ~/dev/sora/skills/
```

Then add `webdev-pipeline` to `agent.skills` in `~/dev/sora/sora/config.yaml`, and restart sora. Leave the underscored `_pipeline-engine` out of `agent.skills`; it is a shared library, not a skill. The manifest reaches the code repo through `WEBDEV_ROOT`, so nothing else changes.
