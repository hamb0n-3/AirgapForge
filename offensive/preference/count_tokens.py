#!/usr/bin/env python3
# pip install mlx-lm datasets

import json, pathlib, sys
from mlx_lm import load

MODEL_ID = "lmstudio-community/Qwen3-30B-A3B-Thinking-2507-MLX-8bit"
DATA_PATH = pathlib.Path("data/train.jsonl")  # adjust if needed
MAX_SEQ_LEN = 4096  # set to whatever you train with

def main():
    print(f"Loading tokenizer for {MODEL_ID} ...")
    _, tok = load(MODEL_ID)  # only need tokenizer

    if not DATA_PATH.exists():
        print(f"ERROR: {DATA_PATH} not found")
        sys.exit(1)

    line = DATA_PATH.read_text(encoding="utf-8").splitlines()[0]
    ex = json.loads(line)

    # Try to map typical DPO fields
    system = ex.get("system", "")
    prompt = ex.get("prompt") or ex.get("question") or ""
    chosen = ex.get("chosen") or ""
    rejected = ex.get("rejected") or ""

    if not prompt or not chosen or not rejected:
        print("WARNING: one of prompt/chosen/rejected is empty. DPO will show zeros.")
        print(f"prompt empty? {not bool(prompt)}, chosen empty? {not bool(chosen)}, rejected empty? {not bool(rejected)}")

    def apply_template(resp: str) -> str:
        msgs = []
        if system:
            msgs.append({"role": "system", "content": system})
        msgs.append({"role": "user", "content": prompt})
        msgs.append({"role": "assistant", "content": resp})

        # Force a string output
        if hasattr(tok, "chat_template") and tok.chat_template:
            try:
                return tok.apply_chat_template(
                    msgs,
                    add_generation_prompt=False,
                    tokenize=False   # <-- critical
                )
            except Exception as e:
                print(f"[template warning] {e}; falling back to manual template.")
        # Fallback manual template
        sys_str = f"{system}\n" if system else ""
        return f"{sys_str}<|user|>{prompt}</|user|>\n<|assistant|>{resp}</|assistant|>"

    # prompt-only length (approx)
    prompt_only = f"{system}\n{prompt}" if system else prompt
    prompt_toks = len(tok.encode(prompt_only))

    full_ch = apply_template(chosen)
    full_rj = apply_template(rejected)

    chosen_toks = len(tok.encode(full_ch))
    rejected_toks = len(tok.encode(full_rj))

    print(f"prompt toks (approx): {prompt_toks}")
    print(f"chosen total toks:    {chosen_toks}  ({'OVER' if chosen_toks>MAX_SEQ_LEN else 'OK  '})")
    print(f"rejected total toks:  {rejected_toks}  ({'OVER' if rejected_toks>MAX_SEQ_LEN else 'OK  '})")

    # Rough split: how many tokens are taken by the assistant part only?
    # (Heuristic: build 'up to assistant' separately)
    def assistant_only_tokens(resp: str) -> int:
        msgs = []
        if system:
            msgs.append({"role": "system", "content": system})
        msgs.append({"role": "user", "content": prompt})
        if hasattr(tok, "apply_chat_template"):
            pre = tok.apply_chat_template(msgs, add_generation_prompt=False, tokenize=False)
            return len(tok.encode(resp))
        else:
            return len(tok.encode(resp))

    print(f"chosen (assistant-only) toks ~ {assistant_only_tokens(chosen)}")
    print(f"rejected (assistant-only) toks ~ {assistant_only_tokens(rejected)}")

    if chosen_toks > MAX_SEQ_LEN or rejected_toks > MAX_SEQ_LEN:
        print("\nLikely cause of zero rewards: response got truncated past max_seq_length.")
        print("Fixes:")
        print("- Increase --max-seq-length (e.g., 8192) if supported")
        print("- Shorten prompts; keep only minimal context")
        print("- Split long samples; keep response intact")
        print("- Prefer truncation that keeps the END of the sequence (where the assistant lives)")

if __name__ == "__main__":
    main()
