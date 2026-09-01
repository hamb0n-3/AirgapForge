#!/usr/bin/env python3
# ORPO LoRA train for Qwen3-30B-A3B-Thinking (MLX 8-bit). --preview samples and exits.
# Data: JSONL under ./data with prompt/chosen/rejected (optional system).
import argparse
import json
import sys
from pathlib import Path
import subprocess

MODEL_ID = "lmstudio-community/Qwen3-30B-A3B-Thinking-2507-MLX-8bit"

DATA_DIR = Path("./data")
ADAPTER_SAVE_DIR = Path("./adapters")

BATCH_SIZE = 2
GRAD_ACCUM = 8
ITERS = 300
LEARNING_RATE = 1e-4
BETA = 0.1
REWARD_SCALING = 1.0
MAX_SEQ_LEN = 4096
STEPS_PER_REPORT = 10
STEPS_PER_EVAL = 50
SAVE_EVERY = 100

LORA_RANK = 16
LORA_DROPOUT = 0.05
LORA_SCALE = 32

PREVIEW_MAX_TOKENS = 512
PREVIEW_TEMP = 0.2


def any_dataset_present(data_dir: Path) -> bool:
    return any(data_dir.glob("*.jsonl")) or any(data_dir.glob("*.json")) or any(data_dir.glob("*.parquet"))


def preview_prompt_from_data_or_default() -> tuple[str, str]:
    default_system = "You are a strict security assistant. Provide safe fixes, never exploit code."
    default_prompt = "Explain why using eval() on user input is unsafe and provide a safe alternative."

    jl = next(iter(DATA_DIR.glob("*.jsonl")), None)
    if jl:
        try:
            with open(jl, "r", encoding="utf-8") as f:
                line = f.readline().strip()
                if line:
                    ex = json.loads(line)
                    return ex.get("system", default_system), ex.get("prompt", default_prompt)
        except Exception:
            pass
    return default_system, default_prompt


def do_preview():
    latest = None
    try:
        files = [p for p in ADAPTER_SAVE_DIR.glob("*.safetensors")]
        latest = max(files, key=lambda p: p.stat().st_mtime) if files else None
    except Exception:
        latest = None

    adapter_arg = []
    if latest is not None:
        adapter_arg = ["--adapter-path", str(ADAPTER_SAVE_DIR)]
        print(f"[preview] Using adapters from: {ADAPTER_SAVE_DIR} (latest file: {latest.name})")
    else:
        print("[preview] No adapter found; using base model.")

    system, prompt = preview_prompt_from_data_or_default()

    cmd = [
        sys.executable, "-m", "mlx_lm.generate",
        "--model", MODEL_ID,
        "--max-tokens", str(PREVIEW_MAX_TOKENS),
        "--temperature", str(PREVIEW_TEMP),
    ] + adapter_arg + ["--prompt", f"<|system|>\n{system}\n<|user|>\n{prompt}\n<|assistant|>"]

    print("[preview] Launching generation:\n" + " ".join(cmd) + "\n")
    proc = subprocess.Popen(cmd)
    proc.wait()


def do_train():
    if not DATA_DIR.exists() or not any_dataset_present(DATA_DIR):
        print(f"Error: no dataset files found under {DATA_DIR} (expect *.jsonl / *.json / *.parquet).")
        sys.exit(1)
    ADAPTER_SAVE_DIR.mkdir(parents=True, exist_ok=True)

    lora_params = {"rank": LORA_RANK, "dropout": LORA_DROPOUT, "scale": LORA_SCALE}

    cmd = [
        sys.executable, "-m", "mlx_lm_lora.train",
        "--model", MODEL_ID,
        "--train",
        "--train-mode", "orpo",
        "--train-type", "lora",
        "--data", str(DATA_DIR),
        "--batch-size", str(BATCH_SIZE),
        "--gradient-accumulation-steps", str(GRAD_ACCUM),
        "--iters", str(ITERS),
        "--learning-rate", str(LEARNING_RATE),
        "--beta", str(BETA),
        "--reward-scaling", str(REWARD_SCALING),
        "--adapter-path", str(ADAPTER_SAVE_DIR),
        "--steps-per-report", str(STEPS_PER_REPORT),
        "--steps-per-eval", str(STEPS_PER_EVAL),
        "--save-every", str(SAVE_EVERY),
        "--max-seq-length", str(MAX_SEQ_LEN),
        "--load-in-8bits",
        "--lora-parameters", json.dumps(lora_params),
        "--prompt-feature", "prompt",
        "--chosen-feature", "chosen",
        "--rejected-feature", "rejected",
        "--system-feature", "system",
    ]

    print("[train] Launching MLX-LM-LORA training:\n" + " ".join(cmd) + "\n")
    proc = subprocess.Popen(cmd)
    try:
        proc.wait()
    except KeyboardInterrupt:
        print("\n[train] Interrupted; terminating...")
        proc.terminate()
        proc.wait()

    if proc.returncode == 0:
        print("\n[train] Training completed.")
    else:
        print(f"\n[train] Training failed with code {proc.returncode}.")


def main():
    parser = argparse.ArgumentParser(description="ORPO train (hard-coded), with optional --preview.")
    parser.add_argument("--preview", action="store_true", help="Preview a test prompt/response and exit (no training).")
    args = parser.parse_args()

    if args.preview:
        do_preview()
        return

    do_train()


if __name__ == "__main__":
    main()
