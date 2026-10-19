# Pipeline manifest schema

The engine (`pipeline.py`) and gate (`preflight.py`) are domain-agnostic. Everything a
domain needs, its repo location, its scripts, its stages, its hardware floors, lives in a
per-domain `manifest.yaml`. To add a domain (medical, legal, whatever), copy an existing
manifest and rewrite the stages. You do not touch the engine.

## Why a manifest and not more skills-with-code

The pipeline shape (collect, prepare, train, evaluate) is the same across domains, but the
scripts, data sources, and benchmarks are not. Keeping the shape in one engine and the
specifics in a manifest means a fix to backgrounding, resumability, or the preflight gate
lands once for every domain, and a new domain is a YAML file plus a short SKILL.md.

## Top-level fields

```yaml
name: airgapforge-cyber        # used in logs (~/.pipeline/<name>/logs) and headers
root_env: AIRGAPFORGE_ROOT     # env var that overrides repo_root (each domain can have its own)
repo_root: /abs/path/to/repo   # default repo location; the repo the commands run against
venv_env: VENV_PYTHON          # env var that overrides venv_python
venv_python: ~/.venv/bin/python
env: {WANDB_MODE: offline}     # set for every step, only if not already in the environment
imports: [mlx_lm]              # interpreter modules preflight must find (FAIL if missing)
hardware:
  helper_ram_gb: 16            # RAM floor for any stage whose tiers include "helper"
  min_ram_gb: {"30B": 32}      # RAM floor per model tier
  min_free_gb: {base: 10, per_tier: {"30B": 12}}   # free-disk floor: base + each selected tier
```

Paths in commands are relative to `repo_root` (the engine runs each step with that cwd),
so a manifest can live in `~/dev/sora/skills/` and still drive a repo elsewhere. Repo
paths are assumed free of spaces.

## Stages

`stages` is an ordered list. Each stage has a `name`, optional `cwd` (relative to
`repo_root`, for stages whose scripts import a sibling like `config.py`), `tiers` (which
hardware floors apply; `helper` or model-tier names), `requires` (paths that must exist, or
preflight FAILs), and `wants` (paths that WARN if missing). A stage holds `steps` and/or a
`matrix`.

### Steps

- **run**: `{label, run, produces?, cwd?, passthrough?}`. `run` is a command string;
  `${py}`, `${root}`, and any gate-set variable are substituted, then it is split into argv
  and run without a shell. `produces` is an output dir; if it is already populated the step
  is skipped unless `--force`. `passthrough: true` appends the engine's trailing `-- <args>`
  to this step (used to forward flags to an eval runner).
- **gate**: `{gate: {name, prefer_if_populated, default, sets, pause_message}}`. If
  `prefer_if_populated` exists it sets `sets` to that and continues; else if the engine was
  run with `--pause` it prints `pause_message` and stops the stage (for a manual step); else
  it sets `sets` to `default` and continues. This is how an optional human step is skipped by
  default and opted into with `--pause`.

### Matrix

`{matrix: {list_cmd, fields, select_field?, skip_if_missing?, run, append_when?}}`. `list_cmd`
emits tab-separated rows; each becomes a dict keyed by `fields`. `--only <value>` keeps the
row whose `select_field` equals it. `skip_if_missing: {field, contains}` skips a row when
that field's value contains the substring and the path it names is not populated (for bases
that must be provisioned first). `run` is the per-row command with `${field}` substituted, and
each `append_when: {field, equals, args}` appends args when a field matches (e.g. a lower
learning rate for one class of model).

## Engine flags

`pipeline.py --manifest <m> --stage <s>` plus: `--only`, `--pause`, `--tier`, `--force`,
`--foreground` (default backgrounds to a log), `--no-preflight`, `--dry-run`, and trailing
`-- <args>` forwarded to the passthrough step. `preflight.py --manifest <m> --stages <csv|all>
--tier <csv|all>` runs the gate on its own.
