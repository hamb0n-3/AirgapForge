# AirgapForge

A research project on building cyber-security instruction data and fine-tuning
LLMs on it, run **entirely on-device** (Apple Silicon / MLX). It is a local
reimplementation of and builds on **CyberLLMInstruct** (ElZemity, Arief & Li,
*A Pseudo-malicious Dataset Revealing Safety-Performance Trade-offs in Cyber
Security LLM Fine-tuning*, ACM AISec '25, arXiv:2503.09334); that work should
be cited by anything derived from this repository.

> **Dual-use notice.** The `offensive/` branch deliberately builds
> exploitation-oriented training data and fine-tunes de-aligned (abliterated)
> models. It is intended for defensive security research and safety
> measurement. No exploit payloads, offensive model weights, or capability/
> safety benchmarks are released.

## Layout

```
AirgapForge/
├── dataset_creation/   Defensive pipeline: collect -> filter -> structure ->
│                       human review -> assemble -> chat-template/format.
│                       Fully local MLX inference; no external API.
├── offensive/          Offensive branch (dual-use).
│   ├── data/           Exploitation SFT dataset (train/valid/test.jsonl + meta),
│   │                   synthesized over CyberNative/Code_Vulnerability_Security_DPO.
│   ├── generation/     Scripts that synthesize the SFT data with a local model.
│   └── preference/     DPO/ORPO setup (defensive: prefer safe rewrites) + Archive/.
├── training/           Fine-tuning configs (LoRA/MLX) for aligned, abliterated,
│                       and gpt-oss bases.
│   └── adapters/       Per-run adapter configs (config only; weights not kept).
├── telemetry/          Recovered training telemetry (no eval metrics).
│   ├── runs/           Exported W&B runs (run.json + history.jsonl per run).
│   ├── results/        W&B output logs / metadata / summary.
│   └── download_wandb.py
├── paper/              LaTeX write-up (dual-use safety study). Self-contained;
│                       builds with `make`. See paper/README.md.
├── README.md · LICENSE · requirements.txt
```

## The defensive pipeline (`dataset_creation/`)

Seven implemented stages, each writing to its own (git-ignored) output
directory so the run can be inspected and resumed. Every LLM-dependent stage
runs locally via MLX.

| Script | Stage |
| --- | --- |
| `1_data_collector.py`         | Collect from NVD, OpenCVE, MITRE ATT&CK, CAPEC, vendor advisories, arXiv, CTF |
| `2_data_filter.py`            | Two-pass filter: parallel rule-based gate + local-LLM enrichment |
| `3_data_structurer.py`        | Source-specific instruction/response synthesis (resumable) |
| `4_manual_reviewer.py`        | Interactive human review with a quality rubric |
| `5_final_assembler.py`        | Deduplicate, validate, export (JSON/JSONL/CSV/HF/MLX) |
| `6_chat_template_converter.py`| Convert to 8 chat templates + train/val/test split |
| `7_format_dataset.py`         | Deterministic Markdown normalisation |
| `utils.py`                    | JSON extraction + `BenchmarkTracker` |

> **Two stages from the original design are intentionally not implemented:** a
> domain classifier (domain labels are instead derived by keyword matching) and
> a security-alignment injector (no automatic alignment step is applied). The
> latter is directly relevant to the offensive branch, which proceeds without it.

## Setup

Apple Silicon + Python 3.8+. See `requirements.txt`. API keys for the
collectors go in a `.env` (see `1_data_collector.py`): `GITHUB_TOKEN`,
`NVD_API_KEY`, `OPENCVE_EMAIL`/`OPENCVE_PASSWORD`, etc.

```sh
pip install -r requirements.txt
python3 dataset_creation/1_data_collector.py --sources all   # then 2-7 in order
```

## Paper

`paper/` contains the write-up and regenerates all of its tables/figures from
the data in this repo (`offensive/data/`, `training/`, `telemetry/runs/`).
Build with `cd paper && make`.

## Not in version control

The following are gitignored and regenerable, not shipped with the repo:

- `offensive/data/*.jsonl`, `offensive/preference/**/*.jsonl` — the exploit
  datasets (regenerate from the public CyberNative source via
  `offensive/generation/`). Only the small `code_audit_sft.meta.jsonl` provenance
  file is kept.
- Model weights and converted bases — `training/adapters/eval_*/*.safetensors`,
  `eval/data/mlx_converted/` (re-train via `eval/train_adapters.py`; re-download
  bases via `eval/download_models.py`).
- `eval/raw/` — raw model generations (dual-use); only the `eval/results/`
  aggregate metrics are committed.
- Downloaded benchmarks (`eval/data/`), Python virtual environments, and paper
  build artifacts.
