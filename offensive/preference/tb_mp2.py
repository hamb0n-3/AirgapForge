#!/usr/bin/env python3
# ORPO LoRA train for Qwen3-30B-A3B (MLX 8-bit). --preview samples and exits.
import argparse
import json
import sys
import math
from pathlib import Path
import subprocess

MODEL_ID = "lmstudio-community/Qwen3-30B-A3B-Thinking-2507-MLX-8bit"

DATA_DIR = Path("./data")
ADAPTER_SAVE_DIR = Path("./adapters")

NUM_LAYERS = 48

BATCH_SIZE = 2
GRAD_ACCUM = 8
ITERS = 300
LEARNING_RATE = 1e-4
BETA = 0.5
REWARD_SCALING = 1.0
MAX_SEQ_LEN = 4096
STEPS_PER_REPORT = 5
STEPS_PER_EVAL = 10
SAVE_EVERY = 100

LORA_RANK = 16
LORA_DROPOUT = 0.05
LORA_SCALE = 32

PREVIEW_MAX_TOKENS = 4096
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
        sys.executable, "-m", "mlx_lm","generate",
        "--model", MODEL_ID,
        "--max-tokens", str(PREVIEW_MAX_TOKENS),
        "--temp", str(PREVIEW_TEMP),
    ] + adapter_arg + ["--prompt", f"<|system|>\n{system}\n<|user|>\n{prompt}\n<|assistant|>"]

    print("[preview] Launching generation:\n" + " ".join(cmd) + "\n")
    proc = subprocess.Popen(cmd)
    proc.wait()

def load_local_preference_dataset(tokenizer):
    try:
        from datasets import Dataset as HFDataset
    except ImportError as e:
        print("Error: the `datasets` library is required for training.\n"
              "Install it with `pip install datasets`.")
        raise e

    examples = []
    for path in DATA_DIR.glob("*.jsonl"):
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    ex = json.loads(line)
                except Exception:
                    continue
                if isinstance(ex, dict) and all(k in ex for k in ("prompt", "chosen", "rejected")):
                    examples.append(ex)
    for path in DATA_DIR.glob("*.json"):
        try:
            with path.open("r", encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, list):
                for ex in data:
                    if isinstance(ex, dict) and all(k in ex for k in ("prompt", "chosen", "rejected")):
                        examples.append(ex)
        except Exception:
            continue
    if not examples:
        print(f"Error: no valid preference records found under {DATA_DIR} (expect 'prompt', 'chosen', 'rejected').")
        sys.exit(1)
    return HFDataset.from_list(examples)

def do_train():
    if not DATA_DIR.exists() or not any_dataset_present(DATA_DIR):
        print(f"Error: no dataset files found under {DATA_DIR} (expect *.jsonl / *.json).")
        sys.exit(1)
    ADAPTER_SAVE_DIR.mkdir(parents=True, exist_ok=True)

    try:
        from mlx_lm import load as mlx_load
        from mlx_lm_lora.trainer.orpo_trainer import ORPOTrainingArgs, train_orpo
        from mlx_lm_lora.trainer.datasets import PreferenceDataset, CacheDataset
        from mlx_lm.tuner.utils import linear_to_lora_layers, print_trainable_parameters
        from mlx_lm.utils import save_config
        import mlx.optimizers as optim
        from mlx_lm.tuner.callbacks import TrainingCallback
    except ImportError as e:
        print("Error: required packages missing. Please install `mlx-lm`, `mlx-lm-lora` and `datasets`.\n"
              "Run: pip install mlx-lm mlx-lm-lora datasets")
        raise e

    print(f"[train] Loading base model '{MODEL_ID}'...")
    model, tokenizer = mlx_load(MODEL_ID)
    model.freeze()

    lora_params = {"rank": LORA_RANK, "dropout": LORA_DROPOUT, "scale": LORA_SCALE}
    print(f"[train] Converting {NUM_LAYERS} layers to LoRA with parameters: {lora_params}")
    linear_to_lora_layers(model=model, num_layers=NUM_LAYERS, config=lora_params, use_dora=False)
    print_trainable_parameters(model)

    print("[train] Loading local preference dataset...")
    raw_dataset = load_local_preference_dataset(tokenizer)
    train_dataset, valid_dataset = raw_dataset.train_test_split(test_size=0.01, seed=42).values()
    train_set = PreferenceDataset(train_dataset, tokenizer)
    valid_set = PreferenceDataset(valid_dataset, tokenizer)

    adapter_path = ADAPTER_SAVE_DIR
    config_data = {
        "lora_parameters": lora_params,
        "num_layers": NUM_LAYERS,
    }
    save_config(config_data, adapter_path / "adapter_config.json")

    num_samples = len(train_set)
    epochs = 2
    batches_per_epoch = max(1, math.ceil(num_samples / BATCH_SIZE))
    calculated_iters = epochs * batches_per_epoch
    iters = max(ITERS, calculated_iters)
    print(f"[train] Calculated {iters} iterations from {epochs} epochs (dataset size: {num_samples}, batch size: {BATCH_SIZE})")

    training_args = ORPOTrainingArgs(
        batch_size=BATCH_SIZE,
        iters=iters,
        val_batches=1,
        steps_per_report=STEPS_PER_REPORT,
        steps_per_eval=STEPS_PER_EVAL,
        steps_per_save=SAVE_EVERY,
        adapter_file=adapter_path,
        max_seq_length=MAX_SEQ_LEN,
        grad_checkpoint=True,
        beta=BETA,
        reward_scaling=REWARD_SCALING,
    )

    opt = optim.AdamW(learning_rate=LEARNING_RATE)

    print("[train] Starting ORPO training with MLX backend...")
    train_orpo(
        model=model,
        args=training_args,
        optimizer=opt,
        train_dataset=CacheDataset(train_set),
        val_dataset=CacheDataset(valid_set),
        training_callback=TrainingCallback(),
    )

    print("\n[train] Training complete. To fuse adapters into a standalone MLX model, run:\n"
    "    python -m mlx_lm fuse --model lmstudio-community/Qwen3-30B-A3B-MLX-8bit \\\n"
    "        --adapter-file ./adapters/adapters.safetensors --save-path ./lora_fused_model\n"
    "(Use 'python -m mlx_lm fuse ...' or 'mlx_lm fuse ...' depending on your environment.)\n")


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Fine‑tune the Qwen3‑30B‑A3B model via ORPO with LoRA. "
            "Use --preview to sample a response from the current model."
        )
    )
    parser.add_argument("--preview", action="store_true", help="Preview a test prompt/response and exit (no training).")
    args = parser.parse_args()

    if args.preview:
        do_preview()
        return

    do_train()


if __name__ == "__main__":
    main()
