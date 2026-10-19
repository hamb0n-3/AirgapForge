#!/usr/bin/env python3
"""Domain-agnostic pipeline engine. Runs one stage described by a YAML manifest.
A stage is a list of steps (run / gate / matrix) plus an optional model matrix; the
engine gates on preflight, applies the manifest's env, skips finished steps, forwards
trailing '-- <args>' to the stage's passthrough step, and backgrounds long stages to a
log. Everything domain-specific lives in the manifest. See schema.md."""
import argparse
import os
import shlex
import subprocess
import sys
from datetime import datetime
from string import Template
from types import SimpleNamespace

HERE = os.path.dirname(os.path.abspath(__file__))
PREFLIGHT = os.path.join(HERE, "preflight.py")
sys.path.insert(0, HERE)
import common


def build_argv(template, variables):
    return [Template(tok).safe_substitute(variables) for tok in shlex.split(template)]


def relnote(cwd, root):
    return "" if cwd == root else f"  (cwd={os.path.relpath(cwd, root)})"


def show(argv, cwd, ctx):
    print("  →", " ".join(shlex.quote(a) for a in argv) + relnote(cwd, ctx.root), flush=True)


def do_run(step, ctx, stage_cwd, extra):
    cwd = os.path.normpath(os.path.join(ctx.root, step["cwd"])) if step.get("cwd") else stage_cwd
    produces = step.get("produces")
    out = os.path.join(ctx.root, produces) if produces else None
    label = step.get("label", "run")
    if out and (os.path.isfile(out) or common.populated(out)) and not ctx.force:
        print(f"• skip {label}: {produces} already present (use --force)")
        return
    argv = build_argv(step["run"], ctx.vars)
    if step.get("passthrough") and extra:
        argv += extra
    print(f"• {label}")
    show(argv, cwd, ctx)
    if not ctx.dry:
        subprocess.run(argv, cwd=cwd, env=ctx.env, check=True)


def do_gate(gate, ctx, pause):
    prefer = gate.get("prefer_if_populated")
    if prefer and common.populated(os.path.join(ctx.root, prefer)):
        ctx.vars[gate["sets"]] = prefer
        print(f"• gate {gate.get('name', '')}: using {prefer}")
        return True
    if pause:
        print()
        for line in gate.get("pause_message", []):
            print(Template(line).safe_substitute(ctx.vars))
        return False
    ctx.vars[gate["sets"]] = gate["default"]
    print(f"• gate {gate.get('name', '')}: using {gate['default']} (pass --pause to stop here)")
    return True


def do_matrix(matrix, ctx, cwd, only):
    try:
        out = subprocess.check_output(build_argv(matrix["list_cmd"], ctx.vars),
                                      cwd=cwd, env=ctx.env).decode()
    except Exception as e:
        if ctx.dry:
            print(f"• (could not list matrix items: {e})")
            return
        raise
    items = [dict(zip(matrix["fields"], ln.split("\t"))) for ln in out.splitlines() if ln]
    field = matrix.get("select_field")
    if only and field:
        items = [it for it in items if it.get(field) == only]
        if not items:
            sys.exit(f"unknown {field} {only}")
    miss = matrix.get("skip_if_missing")
    for it in items:
        variables = {**ctx.vars, **it}
        if miss and miss.get("contains", "") in it.get(miss["field"], "") \
                and not common.populated(it[miss["field"]]):
            print(f"• skip {it.get(field, '?')}: {os.path.relpath(it[miss['field']], ctx.root)} "
                  "not present; provision it, then re-run")
            continue
        argv = build_argv(matrix["run"], variables)
        for rule in matrix.get("append_when", []):
            if it.get(rule["field"]) == rule.get("equals"):
                argv += shlex.split(rule["args"])
        print(f"• {it.get(field, 'item')}")
        show(argv, cwd, ctx)
        if not ctx.dry:
            subprocess.run(argv, cwd=cwd, env=ctx.env, check=True)


def run_stage(mf, name, ctx, args, extra):
    stages = {s["name"]: s for s in mf["stages"]}
    if name not in stages:
        sys.exit(f"unknown stage {name}; choices: {', '.join(stages)}")
    stage = stages[name]
    stage_cwd = os.path.normpath(os.path.join(ctx.root, stage.get("cwd", ".")))
    for step in stage.get("steps", []):
        if "gate" in step:
            if not do_gate(step["gate"], ctx, args.pause):
                return
        else:
            do_run(step, ctx, stage_cwd, extra)
    if "matrix" in stage:
        do_matrix(stage["matrix"], ctx, stage_cwd, args.only)


def forward_args(args):
    fwd = ["--manifest", os.path.abspath(args.manifest), "--stage", args.stage, "--tier", args.tier]
    if args.only:
        fwd += ["--only", args.only]
    if args.pause:
        fwd += ["--pause"]
    if args.force:
        fwd += ["--force"]
    return fwd


def background(ctx, args, extra):
    logdir = os.path.expanduser(f"~/.pipeline/{ctx.name}/logs")
    os.makedirs(logdir, exist_ok=True)
    log = os.path.join(logdir, f"{args.stage}-{datetime.now():%Y%m%d-%H%M%S}.log")
    cmd = [ctx.py, os.path.abspath(__file__), *forward_args(args), "--foreground", "--no-preflight"]
    if extra:
        cmd += ["--", *extra]
    with open(log, "w") as f:
        p = subprocess.Popen(cmd, stdout=f, stderr=subprocess.STDOUT,
                             start_new_session=True, cwd=ctx.root)
    print(f"launched '{args.stage}' in the background (pid {p.pid})")
    print(f"log:    {log}")
    print(f"follow: tail -f {log}")


def main():
    argv = sys.argv[1:]
    extra = []
    if "--" in argv:
        i = argv.index("--")
        extra, argv = argv[i + 1:], argv[:i]

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--stage", required=True)
    ap.add_argument("--only", help="matrix: run only the item whose select_field equals this")
    ap.add_argument("--pause", action="store_true", help="stop at a gate's manual step instead of continuing")
    ap.add_argument("--tier", default="all", help="preflight sizing (model tiers or 'all')")
    ap.add_argument("--force", action="store_true", help="re-run steps whose output already exists")
    ap.add_argument("--foreground", action="store_true", help="run inline instead of backgrounding")
    ap.add_argument("--no-preflight", action="store_true", help="skip the readiness gate")
    ap.add_argument("--dry-run", action="store_true", help="print commands without running them")
    args = ap.parse_args(argv)

    mf = common.load_manifest(args.manifest)
    root = common.resolve_root(mf)
    py = common.resolve_python(mf)
    env = {**os.environ}
    for k, v in (mf.get("env") or {}).items():
        env.setdefault(k, str(v))
    ctx = SimpleNamespace(py=py, root=root, env=env, dry=args.dry_run, force=args.force,
                          name=mf.get("name", "pipeline"), vars={"py": py, "root": root})

    if args.dry_run:
        args.foreground = True

    if not args.no_preflight and not args.dry_run:
        rc = subprocess.run([py, PREFLIGHT, "--manifest", os.path.abspath(args.manifest),
                             "--stages", args.stage, "--tier", args.tier]).returncode
        if rc != 0:
            sys.exit("preflight failed; fix the FAIL lines above or pass --no-preflight")

    if not args.foreground:
        background(ctx, args, extra)
        return

    print(f"=== [{ctx.name}] stage {args.stage} start {datetime.now():%Y-%m-%d %H:%M:%S}  root={root} ===")
    try:
        run_stage(mf, args.stage, ctx, args, extra)
    except subprocess.CalledProcessError as e:
        sys.exit(f"=== stage {args.stage} FAILED: {' '.join(map(str, e.cmd))} (exit {e.returncode}) ===")
    print(f"=== [{ctx.name}] stage {args.stage} done {datetime.now():%Y-%m-%d %H:%M:%S} ===")


if __name__ == "__main__":
    main()
