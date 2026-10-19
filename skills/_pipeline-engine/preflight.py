#!/usr/bin/env python3
"""Manifest-driven readiness check: hardware, interpreter imports, and each stage's
declared requires (FAIL if missing) and wants (WARN). Prints a table and exits
non-zero when a requested stage cannot run. Domain specifics come from the manifest."""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--stages", default="all", help="comma list of stage names or 'all'")
    ap.add_argument("--tier", default="all", help="comma list of model tiers or 'all'")
    args = ap.parse_args()

    mf = common.load_manifest(args.manifest)
    root = common.resolve_root(mf)
    py = common.resolve_python(mf)
    hw = mf.get("hardware", {})
    ram_map = hw.get("min_ram_gb", {})
    free_map = hw.get("min_free_gb", {})

    names = [s["name"] for s in mf["stages"]]
    sel_names = names if args.stages == "all" else [s.strip() for s in args.stages.split(",")]
    sel_stages = [s for s in mf["stages"] if s["name"] in sel_names]

    req_tiers = list(ram_map) if args.tier == "all" else [t.strip() for t in args.tier.split(",")]
    used, uses_helper = set(), False
    for s in sel_stages:
        for t in s.get("tiers", []):
            if t == "helper":
                uses_helper = True
            else:
                used.add(t)
    tiers = [t for t in req_tiers if t in used]

    rows = []
    def add(scope, check, ok, detail, warn_only=False):
        rows.append((scope, check, "OK" if ok else ("WARN" if warn_only else "FAIL"), detail))

    add("host", "Apple Silicon", sys.platform == "darwin" and os.uname().machine == "arm64",
        f"{sys.platform}/{os.uname().machine}", warn_only=True)
    add("host", "repo root exists", os.path.isdir(root), root)
    for mod in mf.get("imports", []):
        add("host", f"python: {mod}", common.can_import(py, mod), py)

    ram = common.total_ram_gb()
    needed_ram = max([ram_map.get(t, 0) for t in tiers] or [0])
    if uses_helper:
        needed_ram = max(needed_ram, hw.get("helper_ram_gb", 0))
    add("host", "RAM", ram >= needed_ram, f"{ram:.0f} GB, need ~{needed_ram} GB", warn_only=True)

    free = common.free_disk_gb(root)
    needed_disk = free_map.get("base", 0) + sum(free_map.get("per_tier", {}).get(t, 0) for t in tiers)
    add("host", "free disk", free >= needed_disk, f"{free:.0f} GB, need ~{needed_disk} GB",
        warn_only=free >= needed_disk * 0.6)

    for s in sel_stages:
        for p in s.get("requires", []):
            add(s["name"], p, os.path.exists(os.path.join(root, p)), p)
        for p in s.get("wants", []):
            add(s["name"], p, os.path.exists(os.path.join(root, p)), p, warn_only=True)

    width = max(len(f"{s}:{c}") for s, c, _, _ in rows)
    print(f"preflight [{mf.get('name', '?')}]  root={root}")
    print(f"stages={sel_names}  tiers={tiers or ['(none)']}  python={py}\n")
    for scope, check, status, detail in rows:
        print(f"  [{status:<4}] {scope + ':' + check:<{width}}  {detail}")

    failed = [f"{s}:{c}" for s, c, st, _ in rows if st == "FAIL"]
    print()
    if failed:
        print("NOT READY. Fix: " + ", ".join(failed))
        sys.exit(1)
    print("READY (review any WARN lines above).")


if __name__ == "__main__":
    main()
