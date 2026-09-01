import os

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
OFFENSIVE_DATA = os.path.join(ROOT, "offensive", "data")
ADAPTERS_DIR = os.path.join(ROOT, "training", "adapters")
DATA_DIR = os.path.join(HERE, "data")
CONVERTED_DIR = os.path.join(DATA_DIR, "mlx_converted")
CYBERMETRIC_DIR = os.path.join(DATA_DIR, "cybermetric")
RESULTS_DIR = os.path.join(HERE, "results")
RAW_DIR = os.path.join(HERE, "raw")

# Abliterated so it scores exploit content without refusing; override for smoke tests.
JUDGE_MODEL = os.environ.get(
    "EVAL_JUDGE_MODEL", "ailexleon/Huihui-Qwen3.8-27B-abliterated-mlx-8Bit")

GEN_MCQ = {"max_tokens": 256, "temp": 0.0}
GEN_AUDIT = {"max_tokens": 1024, "temp": 0.0}
GEN_JUDGE = {"max_tokens": 256, "temp": 0.0}

MODELS = [
    {"key": "qwen3-4b-aligned",      "repo": "mlx-community/Qwen3-4B-Instruct-2507-4bit",
     "alignment": "aligned",     "tier": "4B"},
    # abliterated repos point at the locally-converted MLX 8-bit base the adapter was
    # trained on; the original HF repo yields degenerate output.
    {"key": "qwen3-4b-abliterated",  "repo": os.path.join(CONVERTED_DIR, "qwen3-4b-abliterated-mlx-8bit"),
     "alignment": "abliterated", "tier": "4B"},
    {"key": "qwen3-30b-aligned",     "repo": "lmstudio-community/Qwen3-30B-A3B-Instruct-2507-MLX-8bit",
     "alignment": "aligned",     "tier": "30B"},
    {"key": "qwen3-30b-abliterated", "repo": os.path.join(CONVERTED_DIR, "qwen3-30b-abliterated-mlx-8bit"),
     "alignment": "abliterated", "tier": "30B"},
    {"key": "gpt-oss-20b-aligned",     "repo": "mlx-community/gpt-oss-20b-MXFP4-Q8",
     "alignment": "aligned",     "tier": "20B"},
    {"key": "gpt-oss-20b-abliterated", "repo": "nightmedia/Huihui-gpt-oss-20b-mxfp4-abliterated-v2-qx86-hi-mlx",
     "alignment": "abliterated", "tier": "20B"},
]

SMOKE_MODEL = {"key": "smoke-qwen3-1.7b", "repo": "mlx-community/Qwen3-1.7B-4bit",
               "alignment": "aligned", "tier": "1.7B"}


def adapter_path_for(key):
    p = os.path.join(ADAPTERS_DIR, f"eval_{key}")
    return p if os.path.isdir(p) else None


def all_configs(include_finetuned=True):
    configs = []
    for m in MODELS:
        configs.append({**m, "variant": "base", "adapter": None})
        if include_finetuned:
            configs.append({**m, "key": m["key"] + "-ft", "base_key": m["key"],
                            "variant": "finetuned",
                            "adapter": os.path.join(ADAPTERS_DIR, f"eval_{m['key']}")})
    return configs
