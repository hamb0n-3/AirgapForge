# Evaluation harness

Local (MLX) capability + safety + quality evaluation of the base and
re-trained fine-tuned models, for the paper's Evaluation section. Runs on
`~/.venv` (mlx, mlx-lm).

## Metrics
- **Capability** — CyberMetric MCQ accuracy (`run_cybermetric.py`). Data fetched
  to `data/cybermetric/` from github.com/CyberMetric.
- **Safety** — refusal rate + attack-success rate on exploit requests
  (`run_safety.py`). The exploit ask is placed in the user turn under a *neutral*
  system prompt so the measure reflects the model's own alignment, not the
  dataset's red-team system prompt.
- **Quality** — LLM-judge score (1–5) of audit answers vs the ground-truth
  reference (`run_quality.py`), under the dataset's intended-use system prompt.

The judge (`lib/judge.py`) is an **abliterated** model
(`ailexleon/Huihui-Qwen3.8-27B-abliterated-mlx-8Bit`) so it scores exploit
content without refusing. It is a scorer, not a subject under test; this choice
is disclosed in the paper. Override with `EVAL_JUDGE_MODEL` (used for smoke tests).

## Model matrix
Defined in `config.py` (`MODELS`): 4B / 30B / 120B, aligned + abliterated, each
evaluated as base and fine-tuned on `offensive/data/` (adapters re-trained since
the originals were lost).

## Run order
```sh
V=$HOME/.venv/bin/python
$V download_models.py                 # 1. fetch base models (~130 GB, resumable)
$V train_adapters.py --all            # 2. re-train LoRA adapters -> training/adapters/eval_*/
$V run_all.py                         # 3. matrix x metrics (resumable; skips done)
# results/results.{json,csv,tex} are the deliverables; results.tex feeds the paper.
```
Useful flags: `run_all.py --base-only` (skip fine-tuned arms until trained),
`--metrics cybermetric`, `--safety-limit 150` (cap judge cost).

## Outputs
- `results/<metric>/<key>.json` — per-config aggregate metrics (committed).
- `results/results.{json,csv,tex}` — joined matrix + LaTeX table (committed).
- `raw/<metric>/<key>.jsonl` — raw generations incl. exploit content
  (**gitignored**, dual-use).

## Smoke test (no big models)
```sh
EVAL_JUDGE_MODEL=mlx-community/Qwen3-1.7B-4bit $V run_cybermetric.py \
  --repo mlx-community/Qwen3-1.7B-4bit --key smoke --n 80 --limit 8
```
