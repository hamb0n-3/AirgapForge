# Paper: AirgapForge 

Dual-use security study: reproducing a cyber-security instruction dataset fully
on-device, and the offensive/de-alignment branch built on the same pipeline.
Builds on and cites CyberLLMInstruct (ElZemity et al., ACM AISec '25,
arXiv:2503.09334).

## Build

Requirements: [`tectonic`](https://tectonic-typesetting.github.io/) and a
Python venv (`.venv/`) with `matplotlib`, `numpy`, `pyyaml`.

```sh
python3 -m venv .venv && ./.venv/bin/pip install matplotlib numpy pyyaml
make            # assets -> pdf -> check  (produces main.pdf)
make arxiv      # main_arxiv.tar.gz, clean-room compilable
```

## Where the numbers come from (all auto-generated, never hand-typed)

- `scripts/gen_wandb.py` — parses recovered training telemetry in
  `../New2/Runs/*/{run.json,history.jsonl}` into `tables/runs.csv` (source of
  truth), `tables/training_cost.tex`, `tables/training_runs.tex`, and loss
  series for figures. **Training metrics only** — no capability/safety eval.
- `scripts/gen_tables.py` — dataset stats from
  `../../../Analysis/CyberLLMInstruct/statistics.json`, offensive split sizes
  from `../New/data/*.jsonl` line counts, hyperparameters from the training
  YAMLs.
- `scripts/gen_figures.py` — domain-coverage bar and loss-curve PDFs.
- `scripts/check_numbers.py` — integrity gate: load-bearing numbers must trace
  to the primary sources, and overclaiming language (that would imply an
  evaluation we did not run) is rejected. `make check` fails the build on
  violation.


