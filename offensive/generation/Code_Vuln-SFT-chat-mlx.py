#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import json
import os
from datetime import datetime
from typing import Dict, List, Optional

from datasets import load_dataset
from tqdm import tqdm

from mlx_lm import load as mlx_load, generate  # type: ignore


SYSTEM_PROMPT = """You are an advanced red team operator detecting vulnerabilites and providing exploitation guidance.
You will be given:
  • A code snippet to audit
Your job:
  1) Identify the vulnerability and where the vulnerability appears
  2) Give step-by-step instructions on how to exploit the vulnerability
  3) Give payload examples
  4) If a short snippet helps, include one (DO NOT paste full programs)
Output format (Markdown):
## Audit Summary
- One or two sentences

## Findings
- Bullet list of issues with brief line/region references if obvious

## Exploitation Steps
- Actionable manual steps

## Payload Examples (auotmated if possible)
```
- Code snippets or commands
```

## References
- Official docs or standards when applicable


"""

USER_TEMPLATE = """Audit the following code for **{vulnerability}**.

**Code to audit (verbatim; may be unsafe):**
<BEGIN_VULNERABLE_CODE>
{code}
<END_VULNERABLE_CODE>

Please follow the required output format and focus on exploitation guidance.
"""


def clean_rejected_text(text: Optional[str]) -> str:
    if not text:
        return ""
    t = text.replace("\r\n", "\n").strip()
    return t


def build_user_prompt(lang: str, vuln: str, rejected_code: str, question: str) -> str:
    return USER_TEMPLATE.format(
        vulnerability=(vuln or "").strip(),
        question=(question or "").strip(),
        lang=(lang or "").strip(),
        code=rejected_code,
    )


def make_chat_messages(lang: str, vuln: str, rejected_code: str, question: str) -> List[Dict[str, str]]:
    user = build_user_prompt(lang, vuln, rejected_code, question)
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user},
    ]


def _build_sampler(temperature: float, top_p: float):
    try:
        from mlx_lm.sample_utils import make_sampler  # type: ignore
        return make_sampler(temp=temperature, top_p=top_p)
    except Exception:
        return None


def run_generation(model, tokenizer, messages: List[Dict[str, str]], max_tokens: int,
                   temperature: float, top_p: float, verbose: bool = False) -> str:
    prompt_text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

    sampler = _build_sampler(temperature, top_p)

    try:
        if sampler is not None:
            out = generate(
                model,
                tokenizer,
                prompt=prompt_text,
                max_tokens=max_tokens,
                sampler=sampler,
                verbose=verbose,
            )
        else:
            out = generate(
                model,
                tokenizer,
                prompt=prompt_text,
                max_tokens=max_tokens,
                verbose=verbose,
            )
    except TypeError:
        out = generate(model, tokenizer, prompt_text, max_tokens=max_tokens)  # type: ignore
    return out if isinstance(out, str) else str(out)


def record_to_chat_sft(
    idx: int,
    lang: str,
    vuln: str,
    question: str,
    rejected: str,
    assistant_text: str,
    include_meta: bool = True,
) -> Dict:
    rec = {
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": build_user_prompt(lang, vuln, rejected, question)},
            {"role": "assistant", "content": assistant_text},
        ]
    }
    if include_meta:
        rec["meta"] = {
            "source_dataset": "CyberNative/Code_Vulnerability_Security_DPO",
            "source_split": "train",
            "row_index": idx,
            "lang": lang,
            "vulnerability": vuln,
        }
    return rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", type=str, default="huihui-ai/Huihui-Qwen3-30B-A3B-Instruct-2507-abliterated",
                    help="HF repo or local path for the model (MLX compatible).")
    ap.add_argument("--split", type=str, default="train", help="Dataset split (default: train).")
    ap.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Max rows to process from the dataset (default: all rows).",
    )
    ap.add_argument("--start", type=int, default=0, help="Start offset in the dataset.")
    ap.add_argument("--langs", type=str, default="",
                    help="Comma-separated allowlist of languages (matches dataset 'lang').")
    ap.add_argument("--output", type=str, default="out/code_audit_sft.jsonl", help="Output JSONL path.")
    ap.add_argument(
        "--meta-output",
        type=str,
        default="",
        help="Output JSONL path for metadata (default: derive from --output).",
    )
    ap.add_argument(
        "--state-file",
        type=str,
        default="",
        help="Path to save/load resume state (default: <output_root>.state.json)",
    )
    ap.add_argument(
        "--resume",
        action="store_true",
        help="Resume from a previous run using the state file.",
    )
    ap.add_argument("--max-tokens", type=int, default=2000, help="Max new tokens per sample.")
    ap.add_argument("--temperature", type=float, default=0.6, help="Sampling temperature (if sampler available).")
    ap.add_argument("--top-p", type=float, default=0.9, help="Nucleus sampling top-p (if sampler available).")
    ap.add_argument("--verbose", action="store_true", help="Verbose model generation.")
    ap.add_argument(
        "--include-meta",
        action="store_true",
        default=True,
        help="Write metadata to a separate JSONL alongside training output.",
    )
    args = ap.parse_args()

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    root, ext = os.path.splitext(args.output)
    meta_output_path = args.meta_output or f"{root}.meta{ext or '.jsonl'}"
    os.makedirs(os.path.dirname(meta_output_path), exist_ok=True)

    state_path = args.state_file or f"{root}.state.json"

    ds = load_dataset("CyberNative/Code_Vulnerability_Security_DPO", split=args.split)

    lang_allow: Optional[set] = None
    if args.langs.strip():
        lang_allow = {x.strip().lower() for x in args.langs.split(",") if x.strip()}

    model, tokenizer = mlx_load(args.model)

    processed = 0
    start = max(args.start, 0)
    total = len(ds)
    if args.limit is None or args.limit <= 0:
        end = total
    else:
        end = min(start + args.limit, total)

    if args.resume and os.path.exists(state_path):
        try:
            with open(state_path, "r", encoding="utf-8") as sf:
                state = json.load(sf)
            saved_split = state.get("split")
            if saved_split and saved_split != args.split:
                print(f"[resume] Warning: overriding split '{args.split}' with saved split '{saved_split}'.")
                ds = load_dataset("CyberNative/Code_Vulnerability_Security_DPO", split=saved_split)
                total = len(ds)
                args.split = saved_split
            saved_output = state.get("output")
            saved_meta = state.get("meta_output")
            if saved_output:
                args.output = saved_output
                root, ext = os.path.splitext(args.output)
                meta_output_path = saved_meta or f"{root}.meta{ext or '.jsonl'}"
            if saved_meta:
                meta_output_path = saved_meta
            start = int(state.get("next_index", start))
            end = int(state.get("end", end))
            processed = int(state.get("processed", 0))
            print(f"[resume] Resuming from index {start} (processed={processed})")
        except Exception as e:
            print(f"[resume] Failed to load state from {state_path}: {e}. Starting fresh.")

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    os.makedirs(os.path.dirname(meta_output_path), exist_ok=True)

    def save_state(current_index: int, last_completed_index: int, processed_count: int):
        st = {
            "dataset": "CyberNative/Code_Vulnerability_Security_DPO",
            "split": args.split,
            "model": args.model,
            "output": args.output,
            "meta_output": meta_output_path,
            "start_original": max(args.start, 0),
            "end": end,
            "last_completed_index": last_completed_index,
            "next_index": current_index,
            "processed": processed_count,
            "langs_allow": sorted(list(lang_allow)) if lang_allow else None,
            "include_meta": bool(args.include_meta),
            "temperature": args.temperature,
            "top_p": args.top_p,
            "max_tokens": args.max_tokens,
            "saved_at": datetime.utcnow().isoformat() + "Z",
        }
        with open(state_path, "w", encoding="utf-8") as sf:
            json.dump(st, sf, ensure_ascii=False, indent=2)
        print(f"[state] Saved resume state to {state_path}")

    file_mode = "a" if (args.resume and os.path.exists(state_path)) else "w"

    last_completed_index = start - 1
    current_index = start
    try:
        with open(args.output, file_mode, encoding="utf-8") as fout, \
             open(meta_output_path, file_mode, encoding="utf-8") as fmeta:
            for i in tqdm(range(start, end), desc="Synthesizing"):
                current_index = i
                ex = ds[i]

                lang = (ex.get("lang") or "").strip()
                if lang_allow and lang.lower() not in lang_allow:
                    continue

                vuln = (ex.get("vulnerability") or "").strip()
                question = (ex.get("question") or "").strip()
                rejected_raw = ex.get("rejected") or ""

                rejected = clean_rejected_text(rejected_raw)

                messages = make_chat_messages(lang=lang, vuln=vuln, rejected_code=rejected, question=question)
                assistant_text = run_generation(
                    model,
                    tokenizer,
                    messages=messages,
                    max_tokens=args.max_tokens,
                    temperature=args.temperature,
                    top_p=args.top_p,
                    verbose=args.verbose,
                ).strip()

                rec = record_to_chat_sft(
                    idx=i,
                    lang=lang,
                    vuln=vuln,
                    question=question,
                    rejected=rejected,
                    assistant_text=assistant_text,
                    include_meta=False,
                )
                fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
                if args.include_meta:
                    meta_obj = {
                        "source_dataset": "CyberNative/Code_Vulnerability_Security_DPO",
                        "source_split": args.split,
                        "row_index": i,
                        "lang": lang,
                        "vulnerability": vuln,
                    }
                    fmeta.write(json.dumps(meta_obj, ensure_ascii=False) + "\n")

                last_completed_index = i
                processed += 1

    except KeyboardInterrupt:
        save_state(current_index=current_index, last_completed_index=last_completed_index, processed_count=processed)
        print("Interrupted by user. State saved. Exiting early.")
        return

    if os.path.exists(state_path):
        try:
            os.remove(state_path)
            print(f"[state] Removed completed state file {state_path}")
        except OSError:
            pass

    meta_note = f" and {processed} meta records to {meta_output_path}" if args.include_meta else ""
    print(f"Done. Wrote {processed} records to {args.output}{meta_note}")


if __name__ == "__main__":
    main()
