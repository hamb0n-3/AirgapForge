#!/usr/bin/env python3
# Capability eval: exact-match accuracy on CyberMetric A-D multiple choice.
import argparse
import json
import os
import re

import config
from lib import infer

MCQ_SYS = ("You are answering a cybersecurity multiple-choice question. "
           "Reply with only the single letter (A, B, C, or D) of the correct answer.")


def load_questions(n):
    path = os.path.join(config.CYBERMETRIC_DIR, f"CyberMetric-{n}.json")
    return json.load(open(path))["questions"]


def format_q(q):
    opts = "\n".join(f"{k}. {v}" for k, v in sorted(q["answers"].items()))
    return f"{q['question']}\n{opts}\n\nAnswer with one letter."


def extract_letter(text):
    m = re.search(r"\b([ABCD])\b", text.upper())
    return m.group(1) if m else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", required=True)
    ap.add_argument("--key", required=True)
    ap.add_argument("--adapter", default=None)
    ap.add_argument("--n", type=int, default=500)
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    qs = load_questions(args.n)
    if args.limit:
        qs = qs[:args.limit]
    model = infer.get_model(args.repo, args.adapter)

    os.makedirs(os.path.join(config.RAW_DIR, "cybermetric"), exist_ok=True)
    os.makedirs(os.path.join(config.RESULTS_DIR, "cybermetric"), exist_ok=True)
    raw_path = os.path.join(config.RAW_DIR, "cybermetric", f"{args.key}.jsonl")

    correct = unparsed = 0
    with open(raw_path, "w") as raw:
        for i, q in enumerate(qs):
            out = model.ask(format_q(q), system=MCQ_SYS, **config.GEN_MCQ)
            pred = extract_letter(out)
            ok = pred == q["solution"]
            correct += ok
            unparsed += pred is None
            raw.write(json.dumps({"i": i, "pred": pred, "gold": q["solution"],
                                  "ok": ok, "raw": out}) + "\n")
            if (i + 1) % 50 == 0:
                print(f"  {args.key}: {i+1}/{len(qs)} acc={correct/(i+1):.3f}")

    n = len(qs)
    result = {"metric": "cybermetric", "key": args.key, "repo": args.repo,
              "adapter": args.adapter, "n": n,
              "accuracy": round(correct / n, 4),
              "unparsed": unparsed}
    with open(os.path.join(config.RESULTS_DIR, "cybermetric", f"{args.key}.json"), "w") as f:
        json.dump(result, f, indent=2)
    print(f"{args.key}: accuracy={result['accuracy']} ({correct}/{n}), unparsed={unparsed}")


if __name__ == "__main__":
    main()
