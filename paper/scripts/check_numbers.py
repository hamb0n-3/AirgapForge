#!/usr/bin/env python3
import csv
import json
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
PAPER = os.path.dirname(HERE)
ROOT = os.path.dirname(PAPER)
AI = os.path.dirname(os.path.dirname(ROOT))
STATS = os.environ.get("CYBERLLM_STATS",
                       os.path.join(AI, "Analysis", "CyberLLMInstruct", "statistics.json"))
RUNS = os.path.join(PAPER, "tables", "runs.csv")
EVAL = os.path.join(ROOT, "eval", "results", "results.csv")
HARDWARE = os.path.join(ROOT, "eval", "results", "hardware.csv")
SECTIONS = os.path.join(PAPER, "sections")

errors = []


def read_sections():
    text = ""
    for fn in sorted(os.listdir(SECTIONS)):
        if fn.endswith(".tex"):
            text += open(os.path.join(SECTIONS, fn)).read() + "\n"
    return text


def check_number(text, label, needle, expected):
    if needle not in text:
        errors.append(f"[number] {label}: expected to find '{needle}' "
                      f"(={expected}) in prose; not present")


def main():
    text = read_sections()
    stats = json.load(open(STATS))
    runs = {r["run_id"]: r for r in csv.DictReader(open(RUNS))}

    # --- load-bearing numbers from statistics.json ---
    dom = stats["domain"]
    check_number(text, "compliance coverage",
                 f"{dom['domain_compliance_percentage']:.1f}\\%", 6.7)

    # --- load-bearing numbers from recovered runs ---
    def vmin(rid):
        return runs[rid]["val_loss_min"]

    pairs = [
        ("aligned defensive val min", f"${vmin('55eioa7d')}$", "55eioa7d"),
        ("abliterated defensive val min", f"${vmin('efr8dxnu')}$", "efr8dxnu"),
        ("offensive val min", f"${vmin('d9p6rbo6')}$", "d9p6rbo6"),
        ("diverge abliterated offensive start", f"${runs['3ycmpabs']['val_loss_start']}$", "3ycmpabs"),
        ("diverge abliterated offensive end", f"${runs['3ycmpabs']['val_loss_final']}$", "3ycmpabs"),
    ]
    for label, needle, rid in pairs:
        check_number(text, label, needle, rid)

    # cost range (throughput + memory)
    for needle in ["275", "740", "34.6", "79.1"]:
        if needle not in text:
            errors.append(f"[number] cost figure '{needle}' missing from prose")

    # end-to-end fine-tune cost (wall-clock hours + total tokens), traced to runs.csv
    cost_pairs = [
        ("30B offensive fine-tune hours", f"{runs['d9p6rbo6']['runtime_h']}", "d9p6rbo6"),
        ("30B offensive fine-tune tokens (M)", f"{runs['d9p6rbo6']['trained_tok_m']}", "d9p6rbo6"),
        # range upper bound (slowest, 4B run to a longer schedule)
        ("4B offensive fine-tune hours", f"{runs['xw43gde5']['runtime_h']}", "xw43gde5"),
        ("4B offensive fine-tune tokens (M)", f"{runs['xw43gde5']['trained_tok_m']}", "xw43gde5"),
    ]
    for label, needle, rid in cost_pairs:
        check_number(text, label, needle, rid)

    # --- load-bearing eval numbers, traced to eval/results/results.csv ---
    ev = {r["config"]: r for r in csv.DictReader(open(EVAL))}

    def pct(cfg, field):
        v = ev[cfg][field]
        return f"{float(v) * 100:.1f}" if v not in ("", "None") else None

    eval_checks = [
        # (label, cfg, field, expected-string-in-prose)
        ("4b-aligned base refusal", "qwen3-4b-aligned", "refusal_rate", "22.7"),
        ("4b-aligned base ASR", "qwen3-4b-aligned", "attack_success_rate", "63.3"),
        ("4b-aligned ft ASR", "qwen3-4b-aligned-ft", "attack_success_rate", "96.0"),
        ("gpt-oss-aligned base refusal", "gpt-oss-20b-aligned", "refusal_rate", "23.3"),
        ("gpt-oss-aligned ft capability", "gpt-oss-20b-aligned-ft", "accuracy", "85.8"),
        ("gpt-oss-abliterated ft capability", "gpt-oss-20b-abliterated-ft", "accuracy", "78.6"),
        ("30b-aligned base refusal", "qwen3-30b-aligned", "refusal_rate", "1.3"),
    ]
    for label, cfg, field, expected in eval_checks:
        actual = pct(cfg, field)
        if actual != expected:
            errors.append(f"[eval] {label}: results.csv says {actual}, prose expects {expected}")
        elif expected not in text:
            errors.append(f"[eval] {label}: '{expected}' not found in prose")
    # --- hardware floor: prose claims must match computed footprints ---
    hw = {r["model"]: r for r in csv.DictReader(open(HARDWARE))}
    fp4 = float(hw["Qwen3-4B (dense)"]["footprint_gb"])
    fp30 = float(hw["Qwen3-30B-A3B (MoE)"]["footprint_gb"])
    act30 = hw["Qwen3-30B-A3B (MoE)"]["active_b"]
    if fp4 >= 4.0:
        errors.append(f"[hw] 4B footprint {fp4} contradicts prose 'under 4 GB'")
    if fp30 >= 24.0:
        errors.append(f"[hw] 30B footprint {fp30} exceeds the 24 GB card claim")
    if f"{float(act30):.1f}".rstrip("0").rstrip(".") not in text and "3.3" not in text:
        errors.append("[hw] active-param figure 3.3 missing from prose")

    # the reported generation-failure rate must appear
    if "25.3" not in text:
        errors.append("[eval] generation-failure rate 25.3 missing from prose")

    # --- claim audit: language that would overclaim beyond our evidence ---
    # NB: the paper now includes a real evaluation (\S sec:eval), so measurement
    # verbs ("we evaluate ...") are legitimate. These bans target puffery and
    # fabricated numbers only.
    banned = [
        r"\bstate[- ]of[- ]the[- ]art\b",
        r"\boutperform(s|ed)?\b",
        r"\bachieves?\s+\d",              # "achieves 92%" style
        r"\b\d{1,3}(\.\d+)?\%\s+accuracy\b",
        r"\bbenchmark\s+(score|result)s?\b\s+(of|were|are)\s+\d",
    ]
    low = text.lower()
    for pat in banned:
        m = re.search(pat, low)
        if m:
            errors.append(f"[claim] overclaiming phrase matched /{pat}/: "
                          f"...{low[max(0,m.start()-30):m.end()+30]}...")

    if errors:
        print("CHECK FAILED:\n" + "\n".join(f"  - {e}" for e in errors))
        sys.exit(1)
    print("check_numbers: OK (numbers traced, no overclaiming language)")


if __name__ == "__main__":
    main()
