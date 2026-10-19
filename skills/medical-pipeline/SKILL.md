---
name: medical-pipeline
description: "Train general medical knowledge into a local model on Apple Silicon: build an educational medical instruction dataset from authoritative public sources, train LoRA adapters with MLX, and score them with a local eval harness. Runs on the shared pipeline engine from this domain's manifest. For educational information only, not diagnosis or treatment."
version: 0.1.0
author: Tristan
license: MIT
platforms: [macos]
metadata:
  tags: [Training, MLX, LoRA, Dataset, Evaluation, Medical, Education]
  related_skills: [webdev-pipeline, airgapforge-cyber]
---

# Medical pipeline

> Educational scope. This trains a model for general, educational health information, not for diagnosis or treatment. The chat template carries a system prompt saying the model is not a substitute for professional medical advice and that people should consult a qualified clinician and seek emergency care when appropriate. Draw only on authoritative public sources, and keep content educational rather than personalized clinical advice.

This is the medical instance of the reusable pipeline. The domain-agnostic engine lives in `_pipeline-engine/`, and everything specific to this domain (its scripts, sources, model matrix, hardware floors) is described in `manifest.yaml`. The engine gates on preflight, sets `WANDB_MODE=offline`, skips finished steps, runs the model matrix, and backgrounds long stages to a log. You fill in the manifest and write the scripts it names; you do not edit the engine.

It has three stages: `data`, `train`, `eval`. There is no offensive stage. Eval is capability (MedQA-style MCQ) plus quality (LLM judge).

## Paths

The engine, this manifest, and the medical-forge code repo are three separate things. The first two get migrated into `~/dev/sora/skills/`; the code repo is named by the manifest's `repo_root` (override with `MEDICAL_ROOT`), so it can live anywhere.

```sh
PY=~/.venv/bin/python                                # has mlx and pyyaml; runs the engine
ENG=~/dev/sora/skills/_pipeline-engine/pipeline.py
MAN=~/dev/sora/skills/medical-pipeline/manifest.yaml
export MEDICAL_ROOT=~/dev/medical-forge              # the repo holding the scripts the manifest names
```

## Setup (once)

Apple Silicon with enough memory for the bases you pick (a 4B fits in 8 GB, an 8B in 16 GB). Then a virtualenv:

```sh
python3 -m venv ~/.venv
~/.venv/bin/pip install -r "$MEDICAL_ROOT"/requirements.txt
```

## Run it

Preflight first; its `requires` failures are the to-do list. Then a stage at a time:

```sh
$PY ~/dev/sora/skills/_pipeline-engine/preflight.py --manifest $MAN --stages all --tier all

$PY $ENG --manifest $MAN --stage data                 # collect, filter, structure, assemble, templates (no review)
$PY $ENG --manifest $MAN --stage data --pause         # stop after structure for the human review
$PY $ENG --manifest $MAN --stage train                # train every base in the matrix; --only <key> for one
$PY $ENG --manifest $MAN --stage eval -- --limit 200  # flags after -- pass straight to run_all.py
```

Each stage backgrounds itself and returns a log path under `~/.pipeline/medical-pipeline/logs/`, so `tail -f` it to watch. Useful flags: `--dry-run`, `--foreground`, `--force`, `--only <key>`, `--pause`, `--tier 8B`.

## Stages

- **data** collects from MedlinePlus, open textbooks (NCBI Bookshelf), PubMed abstracts, and MedQA-style items, then filters, structures into educationally framed instruction/response pairs, optionally pauses for human review, assembles, and writes `chat_templates/` with the educational system prompt. It runs offline (bundled seed corpus) and with no model (deterministic templates). Add `--model <repo>` to the structure step for real local-LLM synthesis, and keep the educational framing.
- **train** trains one LoRA adapter per base in `train/config.py` via `mlx_lm lora`, reading `../chat_templates`.
- **eval** runs `eval/run_all.py`: capability (MCQ over `eval/data/med_quiz.jsonl`) and quality (LLM judge over `eval/data/med_quality.jsonl`), for the base and fine-tuned arms. Deliverables are `eval/results/results.{json,csv}`.

## Filling it in

1. Point `repo_root` (or `MEDICAL_ROOT`) at the medical-forge repo.
2. Expand the sources and seed in `data_pipeline/sources.py` with more authoritative references, keeping content educational.
3. Confirm the 8B repo id in `train/config.py`; 4B is the verified default.
4. Re-run preflight until READY, then run the stages.

For the manifest field reference, see `_pipeline-engine/schema.md`.

## Migrating into sora

```sh
mv "$AIRGAPFORGE_ROOT/skills/_pipeline-engine"   ~/dev/sora/skills/   # if not already moved
mv "$AIRGAPFORGE_ROOT/skills/medical-pipeline"   ~/dev/sora/skills/
```

Then add `medical-pipeline` to `agent.skills` in `~/dev/sora/sora/config.yaml`, and restart sora. Leave the underscored `_pipeline-engine` out of `agent.skills`; it is a shared library. The manifest reaches the code repo through `MEDICAL_ROOT`, so nothing else changes.
