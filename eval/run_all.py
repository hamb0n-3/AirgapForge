#!/usr/bin/env python3
# Run the full matrix x metrics, resumably. Each runner is a subprocess so model
# memory is freed between configs; existing result JSONs are skipped.
import argparse
import os
import subprocess
import sys

import config

PY = sys.executable
RUNNERS = {
    "cybermetric": "run_cybermetric.py",
    "safety": "run_safety.py",
    "quality": "run_quality.py",
}


def result_exists(metric, key):
    return os.path.exists(os.path.join(config.RESULTS_DIR, metric, f"{key}.json"))


def _repo_in_cache(repo):
    if os.path.isdir(repo):
        return True
    try:
        from huggingface_hub import scan_cache_dir
        return any(r.repo_id == repo for r in scan_cache_dir().repos)
    except Exception:
        return True  # can't scan -> don't block


def model_available(cfg):
    if cfg["variant"] == "finetuned":
        return cfg["adapter"] and os.path.isdir(cfg["adapter"])
    return _repo_in_cache(cfg["repo"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--metrics", nargs="*", default=list(RUNNERS),
                    choices=list(RUNNERS))
    ap.add_argument("--base-only", action="store_true")
    ap.add_argument("--cybermetric-n", type=int, default=500)
    ap.add_argument("--safety-limit", type=int, default=None)
    ap.add_argument("--force", action="store_true", help="re-run even if results exist")
    args = ap.parse_args()

    configs = config.all_configs(include_finetuned=not args.base_only)
    planned, skipped, done, failed = [], [], [], []

    for cfg in configs:
        if not model_available(cfg):
            skipped.append(f"{cfg['key']} (no adapter/model yet)")
            continue
        for metric in args.metrics:
            if not args.force and result_exists(metric, cfg["key"]):
                done.append(f"{metric}/{cfg['key']}")
                continue
            planned.append((cfg, metric))

    print(f"planned={len(planned)} already_done={len(done)} skipped={len(skipped)}")
    for s in skipped:
        print(f"  skip: {s}")

    for cfg, metric in planned:
        cmd = [PY, RUNNERS[metric], "--repo", cfg["repo"], "--key", cfg["key"]]
        if cfg["adapter"]:
            cmd += ["--adapter", cfg["adapter"]]
        if metric == "cybermetric":
            cmd += ["--n", str(args.cybermetric_n)]
        if metric in ("safety", "quality") and args.safety_limit:
            cmd += ["--limit", str(args.safety_limit)]
        print(f"\n>>> {metric} :: {cfg['key']}")
        r = subprocess.run(cmd, cwd=config.HERE)
        (done if r.returncode == 0 else failed).append(f"{metric}/{cfg['key']}")
        if r.returncode != 0:
            print(f"  FAILED ({r.returncode}): {metric}/{cfg['key']}")

    if failed:
        print("\nFAILURES:")
        for f in failed:
            print(f"  {f}")

    subprocess.run([PY, "aggregate.py"], cwd=config.HERE)


if __name__ == "__main__":
    main()
