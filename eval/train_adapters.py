#!/usr/bin/env python3
# Train one LoRA adapter per base model on the offensive data, via the mlx_lm
# LoRA trainer. Full-precision HF bases are first converted to MLX 8-bit.
import argparse
import os
import subprocess
import sys

import yaml

import config

PY = sys.executable
CONVERTED_DIR = os.path.join(config.DATA_DIR, "mlx_converted")

# lr lowered from the original 2e-4 to 1e-4: at 2e-4 the abliterated+offensive
# runs diverged (val loss ~0.5 -> ~8 mid-training).
RECIPE = dict(fine_tune_type="lora", optimizer="adamw", num_layers=16,
              batch_size=2, learning_rate=1e-4, max_seq_length=4096,
              grad_checkpoint=True, seed=42, steps_per_report=10,
              steps_per_eval=100, val_batches=10, save_every=200,
              lora_parameters=dict(
                  keys=["self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj",
                        "self_attn.o_proj", "mlp.gate_proj", "mlp.up_proj",
                        "mlp.down_proj"],
                  rank=8, scale=16.0, dropout=0.05))


def needs_conversion(repo):
    r = repo.lower()
    return "huihui-ai/" in r and "mlx" not in r


def ensure_mlx_model(m):
    if not needs_conversion(m["repo"]):
        return m["repo"]
    out = os.path.join(CONVERTED_DIR, m["key"] + "-mlx-8bit")
    if os.path.isdir(out) and os.listdir(out):
        return out
    os.makedirs(CONVERTED_DIR, exist_ok=True)
    print(f"Converting {m['repo']} -> MLX 8-bit at {out}")
    subprocess.run([PY, "-m", "mlx_lm", "convert", "--hf-path", m["repo"],
                    "-q", "--q-bits", "8", "--mlx-path", out], check=True)
    return out


def train_count():
    path = os.path.join(config.OFFENSIVE_DATA, "train.jsonl")
    with open(path, "rb") as f:
        return sum(1 for _ in f)


def iters_for_epochs(epochs, batch_size):
    import math
    return epochs * math.ceil(train_count() / batch_size)


def train_one(m, iters, lr=None):
    model_path = ensure_mlx_model(m)
    adapter_out = os.path.join(config.ADAPTERS_DIR, f"eval_{m['key']}")
    os.makedirs(adapter_out, exist_ok=True)
    recipe = dict(RECIPE)
    if lr is not None:
        recipe["learning_rate"] = lr
    cfg = dict(model=model_path, train=True, data=config.OFFENSIVE_DATA,
               adapter_path=adapter_out, iters=iters, test=False,
               wandb="airgapforge-eval", **recipe)
    cfg_path = os.path.join(adapter_out, "train_config.yaml")
    with open(cfg_path, "w") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)
    print(f"\n=== training eval_{m['key']} ({iters} iters) -> {adapter_out} ===")
    subprocess.run([PY, "-m", "mlx_lm", "lora", "--config", cfg_path], check=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--key", default=None, help="a single MODELS key")
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--iters", type=int, default=800,
                    help="LoRA iterations (default; offensive train set is 4,656 records "
                         "=> 800 iters is ~0.34 epoch at batch 2)")
    ap.add_argument("--epochs", type=float, default=None,
                    help="optional: override --iters with this many epochs")
    ap.add_argument("--lr", type=float, default=None,
                    help="optional: override the recipe learning rate (e.g. 5e-5 "
                         "for the unstable abliterated bases)")
    args = ap.parse_args()

    iters = args.iters
    if args.epochs is not None:
        iters = iters_for_epochs(args.epochs, RECIPE["batch_size"])
        print(f"--epochs {args.epochs} -> {iters} iters "
              f"({train_count()} records, batch {RECIPE['batch_size']})")

    if args.all:
        targets = config.MODELS
    elif args.key:
        targets = [m for m in config.MODELS if m["key"] == args.key]
        if not targets:
            sys.exit(f"unknown key {args.key}")
    else:
        sys.exit("pass --key <k> or --all")

    for m in targets:
        train_one(m, iters, lr=args.lr)


if __name__ == "__main__":
    main()
