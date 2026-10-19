---
name: airgapforge-cyber
description: "Run the AirgapForge cyber training pipeline on Apple Silicon: build the cyber instruction dataset, synthesize the offensive SFT set, train LoRA adapters with MLX, and score them with the local eval harness. A shared, domain-agnostic engine runs the stages from this domain's manifest, so a stage is one command and other domains reuse the same engine."
version: 2.0.0
author: Tristan (AirgapForge)
license: MIT
platforms: [macos]
metadata:
  tags: [Training, MLX, LoRA, Dataset, Evaluation, Cybersecurity, DualUse]
  related_skills: []
---

# AirgapForge cyber pipeline

> Dual-use research. AirgapForge builds cybersecurity instruction data and fine-tunes local models on it, and the `offensive/` branch deliberately builds exploitation training data and trains de-aligned (abliterated) models. It exists for defensive security research and safety measurement, and it reimplements CyberLLMInstruct (ElZemity, Arief & Li, ACM AISec '25, arXiv:2503.09334), which anything derived from it should cite. Everything runs on-device via MLX, so no data leaves the machine. Run it only in that context.

This is the cyber instance of a reusable pipeline. The domain-agnostic engine lives next door in `_pipeline-engine/`, and everything specific to AirgapForge (its scripts, stages, model matrix, hardware floors) is described in `manifest.yaml`. The engine gates on preflight, sets `WANDB_MODE=offline`, skips finished steps, trains only the model arms whose bases are present, and backgrounds long stages to a log. To add another domain (a medical pipeline, say), you write a new manifest and SKILL.md against the same engine; see `_pipeline-engine/schema.md`.

## Paths

The engine, this manifest, and the AirgapForge checkout are three separate things. The first two get migrated into `~/dev/sora/skills/`; the repo stays where it is, because the manifest names it by absolute path (override with `AIRGAPFORGE_ROOT`). Set three shell variables so the commands stay short:

```sh
PY=~/.venv/bin/python                                # has mlx and pyyaml; runs the engine
ENG=~/dev/sora/skills/_pipeline-engine/pipeline.py   # post-migration path
MAN=~/dev/sora/skills/airgapforge-cyber/manifest.yaml
# before migration, point ENG and MAN at skills/_pipeline-engine and skills/airgapforge-cyber
# inside the AirgapForge repo. To use a different checkout: export AIRGAPFORGE_ROOT=/path/to/repo
```

## Setup (once)

Apple Silicon with enough memory for the tier you pick (the 120B defensive-training config wants about 128 GB, while the 30B eval arms fit in 32 GB). Then a virtualenv with the repo's dependencies:

```sh
python3 -m venv ~/.venv
~/.venv/bin/pip install -r requirements.txt   # brings in mlx, mlx-lm, pyyaml, and the collectors' deps
```

Collector keys go in `$AIRGAPFORGE_ROOT/.env` (see the top of `dataset_creation/1_data_collector.py`): `GITHUB_TOKEN`, `NVD_API_KEY`, `OPENCVE_EMAIL`/`OPENCVE_PASSWORD`. They are only needed for the data stage.

## Run it

Preflight first for the stages and tiers you intend to run, then a stage at a time:

```sh
$PY ~/dev/sora/skills/_pipeline-engine/preflight.py --manifest $MAN --stages all --tier all

$PY $ENG --manifest $MAN --stage data                 # collect, filter, structure, assemble, templates (no review)
$PY $ENG --manifest $MAN --stage data --pause         # stop after structure for the interactive human review
$PY $ENG --manifest $MAN --stage offensive            # synthesize the offensive SFT set
$PY $ENG --manifest $MAN --stage train                # train every present arm; --only <key> for one
$PY $ENG --manifest $MAN --stage eval -- --base-only  # flags after -- pass straight to run_all.py
```

Each stage backgrounds itself and returns a log path under `~/.pipeline/airgapforge-cyber/logs/`, so `tail -f` it to watch. Useful flags: `--dry-run` prints the exact commands without running them, `--foreground` runs inline, `--force` re-runs a step whose output already exists, `--only <key>` limits training to one arm, `--pause` stops at the review gate, and `--tier 30B` sizes the disk and memory check to one tier.

## What each stage does, and its traps

**data** runs the seven `dataset_creation/` scripts. The catch is stage 4: the reviewer reads `domain_classified/`, which no script produces (the domain classifier is unimplemented; labels are keyword-derived), and it expects JSON entries with a `source_data` field. By default the engine skips review and assembles from `structured_data/`; pass `--pause` to stop and do the review, which writes `reviewed_data/`, then re-run and it assembles from there. The result lands in `chat_templates/`; point a defensive training config's `data:` field at the directory holding `train.jsonl`, `valid.jsonl`, and `test.jsonl`.

**offensive** synthesizes the SFT data locally over the public `CyberNative/Code_Vulnerability_Security_DPO` source into `offensive/data/{train,valid,test}.jsonl`, which the training stage reads. These files are gitignored, so regenerate them rather than expecting them in a clean checkout.

**train** trains one LoRA adapter per base in `eval/config.py` on `offensive/data/`, via `eval/train_adapters.py`. The engine sets `WANDB_MODE=offline` (the recipe enables W&B and would otherwise prompt), adds `--lr 5e-5` for the abliterated arms because they diverge at the default rate, and skips any arm whose base is a local `eval/data/mlx_converted/` path that does not exist yet. Those 4B and 30B abliterated bases are not created by any script, so convert them once before training those arms:

```sh
cd "$AIRGAPFORGE_ROOT/eval"
$PY -m mlx_lm convert --hf-path <local base dir or repo> \
     -q --q-bits 8 --mlx-path data/mlx_converted/qwen3-30b-abliterated-mlx-8bit
```

The defensive `training/*.yaml` configs are a separate path (standard `mlx_lm lora` on the stage-1 dataset), not driven by the engine. Set their `data:` field (it ships `[SANITIZED]`) and W&B before running them.

**eval** runs `eval/run_all.py` over the matrix (capability, safety, quality), resumable, skipping done work. It is the only harness that scores these models on cyber knowledge and refusal behavior; sora's own eval benchmarks the sora agent and cannot. Flags after `--` pass through, so `-- --base-only` skips the fine-tuned arms and `-- --metrics cybermetric --safety-limit 150` scopes and caps judge cost. Deliverables are `eval/results/results.{json,csv,tex}`, and `results.tex` feeds the paper.

## Migrating into sora

Move both folders into sora's skills directory (the repo stays put):

```sh
mv "$AIRGAPFORGE_ROOT/skills/_pipeline-engine"   ~/dev/sora/skills/
mv "$AIRGAPFORGE_ROOT/skills/airgapforge-cyber"  ~/dev/sora/skills/
```

Then add `airgapforge-cyber` to `agent.skills` in `~/dev/sora/sora/config.yaml`, and restart sora. `skills_dirs` already includes `~/dev/sora/skills`. The engine finds its own siblings, and the manifest reaches the repo through `AIRGAPFORGE_ROOT`, so nothing else changes. The leading-underscore `_pipeline-engine` is a shared library, not a skill, so leave it out of `agent.skills`.
