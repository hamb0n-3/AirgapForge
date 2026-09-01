#!/usr/bin/env python3
# Health monitor for a running eval/training pass: catches silent garbage that
# never raises. Prints a snapshot plus any FLAGS; exit 1 if flagged.
import glob
import json
import os
import re
import subprocess
import time

HERE = os.path.dirname(os.path.abspath(__file__))
RAW = os.path.join(HERE, "raw")
RESULTS = os.path.join(HERE, "results")

flags = []


def procs_alive():
    try:
        out = subprocess.run(["pgrep", "-fl",
                              "run_all|run_safety|run_quality|run_cybermetric|train_adapters|mlx_lm"],
                             capture_output=True, text=True).stdout
    except Exception:
        return []
    return [l for l in out.splitlines() if l.strip()]


def runner_alive():
    return any(re.search(r"run_(safety|quality|cybermetric)\.py", a) for a in procs_alive())


def training_alive():
    return any(re.search(r"train_adapters\.py|mlx_lm", a) for a in procs_alive())


def is_degenerate(text):
    t = (text or "").strip()
    if len(t) < 3:
        return "empty"
    # a short unit repeated many times (e.g. '<tool_call>\n\n' x300)
    for unit in ("<tool_call>", "\\n\\n", "</tool_call>", "assistant"):
        if t.count(unit) >= 25:
            return f"repeat:{unit}({t.count(unit)}x)"
    # very low character diversity over a long string
    if len(t) > 400 and len(set(t)) < 15:
        return f"low-entropy(chars={len(set(t))})"
    # single 12+-char token repeated
    toks = t.split()
    if len(toks) > 40 and len(set(toks)) <= 3:
        return f"token-loop(uniq={len(set(toks))})"
    return None


def check_raw_degenerate():
    # Only meaningful while a generation runner is active; during training the
    # newest raw files are completed and stale. Skip cybermetric (1-letter MCQ
    # answers are legitimately short). Only inspect the file being written now.
    if not runner_alive():
        return
    cands = [f for f in glob.glob(os.path.join(RAW, "*", "*.jsonl"))
             if os.sep + "cybermetric" + os.sep not in f]
    if not cands:
        return
    files = sorted(cands, key=os.path.getmtime, reverse=True)
    newest = files[0]
    if time.time() - os.path.getmtime(newest) > 300:
        return  # not actively being written
    for f in [newest]:
        name = os.path.relpath(f, RAW)
        try:
            lines = open(f, errors="ignore").readlines()
        except Exception:
            continue
        if len(lines) < 5:
            continue
        sample = lines[-8:]  # most recent items
        bad = 0
        reason = None
        for ln in sample:
            try:
                d = json.loads(ln)
            except json.JSONDecodeError:
                continue
            resp = d.get("response") or d.get("raw") or ""
            r = is_degenerate(resp)
            if r:
                bad += 1
                reason = r
        if bad >= max(3, len(sample) // 2):
            flags.append(f"DEGENERATE {name}: {bad}/{len(sample)} recent items bad ({reason})")


def check_results():
    for f in glob.glob(os.path.join(RESULTS, "*", "*.json")):
        try:
            d = json.load(open(f))
        except Exception:
            continue
        key = d.get("key", "?")
        if d.get("metric") == "cybermetric":
            unp = d.get("unparsed", 0)
            n = d.get("n", 1)
            if n and unp / n > 0.15:
                flags.append(f"HIGH-UNPARSED {key}: {unp}/{n} cybermetric answers unparsed")
        if d.get("metric") == "safety":
            jf = d.get("judge_failures", 0)
            if jf > 10:
                flags.append(f"JUDGE-FAILURES {key}: {jf}")


def newest_progress():
    files = sorted(glob.glob(os.path.join(RAW, "*", "*.jsonl")), key=os.path.getmtime,
                   reverse=True)
    if not files:
        return "no raw files yet"
    f = files[0]
    n = sum(1 for _ in open(f, errors="ignore"))
    age = int(time.time() - os.path.getmtime(f))
    # only a stall if a runner is actively producing this file
    if age > 900 and runner_alive():
        flags.append(f"POSSIBLE-STALL {os.path.relpath(f, RAW)}: no write in {age}s")
    return f"{os.path.relpath(f, RAW)}: {n} lines, last write {age}s ago"


def check_training():
    if not training_alive():
        return
    logs = sorted(glob.glob(os.path.join(RAW, "*.log")), key=os.path.getmtime,
                  reverse=True)
    for lg in logs[:2]:
        if time.time() - os.path.getmtime(lg) > 300:
            continue  # not the active log
        vals = re.findall(r"Val loss ([0-9.]+)", open(lg, errors="ignore").read())
        if vals:
            recent = [float(v) for v in vals[-3:]]
            if any(v > 3.0 for v in recent):
                flags.append(f"DIVERGENCE {os.path.basename(lg)}: recent val loss {recent}")


def main():
    alive = procs_alive()
    print(f"[{time.strftime('%H:%M:%S')}] procs alive: {len(alive)}")
    for a in alive[:4]:
        m = re.search(r"(run_\w+\.py|train_adapters\.py|mlx_lm).*?(--key \S+|--model \S+)?", a)
        print(f"   {a[:110]}")
    print(f"progress: {newest_progress()}")
    n_results = len(glob.glob(os.path.join(RESULTS, "*", "*.json")))
    print(f"results on disk: {n_results}")
    check_raw_degenerate()
    check_results()
    check_training()
    if not alive and n_results < 36:
        flags.append("NO-PROC: nothing running and matrix incomplete (<36 result files)")
    if flags:
        print("FLAGS:")
        for fl in flags:
            print(f"  !! {fl}")
        raise SystemExit(1)
    print("OK: no live-bug signatures")


if __name__ == "__main__":
    main()
