#!/usr/bin/env python3
import argparse
import traceback

from huggingface_hub import snapshot_download

import config


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", nargs="*", default=None,
                    help="download only these config keys")
    args = ap.parse_args()

    models = config.MODELS
    if args.only:
        models = [m for m in models if m["key"] in args.only]

    for m in models:
        print(f"\n=== {m['key']} ({m['tier']}, {m['alignment']}): {m['repo']} ===")
        try:
            path = snapshot_download(m["repo"])
            print(f"  OK -> {path}")
        except Exception as e:
            print(f"  FAILED: {e}")
            traceback.print_exc()


if __name__ == "__main__":
    main()
