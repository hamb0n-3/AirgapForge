#!/usr/bin/env python3
# Quality eval: run each record under the dataset's own audit system prompt, then
# have the abliterated judge score the answer against the reference (1-5).
import argparse
import json
import os

import config
from lib import dataio, infer, judge


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", required=True)
    ap.add_argument("--key", required=True)
    ap.add_argument("--adapter", default=None)
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    records = dataio.load_exploit_testset(limit=args.limit)
    model = infer.get_model(args.repo, args.adapter)

    os.makedirs(os.path.join(config.RAW_DIR, "quality"), exist_ok=True)
    os.makedirs(os.path.join(config.RESULTS_DIR, "quality"), exist_ok=True)
    raw_path = os.path.join(config.RAW_DIR, "quality", f"{args.key}.jsonl")

    scores = []
    gen_fail = 0
    n = len(records)
    with open(raw_path, "w") as raw:
        for i, rec in enumerate(records):
            resp = model.chat(
                [{"role": "system", "content": rec["system"]},
                 {"role": "user", "content": rec["user"]}], **config.GEN_AUDIT)
            if infer.is_degenerate(resp):
                gen_fail += 1
                raw.write(json.dumps({"i": i, "gen_fail": True, "response": resp[:200]}) + "\n")
                continue
            verdict = judge.judge_quality(rec["user"], resp, rec["reference"])
            score = verdict["score"]
            if score is not None:
                scores.append(score)
            raw.write(json.dumps({"i": i, "score": score, "reason": verdict["reason"],
                                  "response": resp}) + "\n")
            if (i + 1) % 25 == 0:
                m = sum(scores) / len(scores) if scores else 0
                print(f"  {args.key}: {i+1}/{n} mean_score={m:.2f} genfail={gen_fail}")

    mean = round(sum(scores) / len(scores), 3) if scores else None
    result = {"metric": "quality", "key": args.key, "repo": args.repo,
              "adapter": args.adapter, "n": n, "scored": len(scores),
              "generation_failures": gen_fail,
              "generation_failure_rate": round(gen_fail / n, 4),
              "mean_score": mean}
    with open(os.path.join(config.RESULTS_DIR, "quality", f"{args.key}.json"), "w") as f:
        json.dump(result, f, indent=2)
    print(f"{args.key}: mean_score={mean} (scored {len(scores)}/{n})")


if __name__ == "__main__":
    main()
