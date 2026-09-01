#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Split a JSONL dataset into valid/test sets, preserving lines verbatim.
import argparse
import json
import math
import os
import random
import sys
from typing import Iterable, Iterator, List, Optional, Sequence, Tuple, Union

def _iter_jsonl(path: str) -> Iterator[str]:
    if path == "-":
        for line in sys.stdin:
            if line.strip():
                yield line
        return
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield line

def _load_slice(path: str, start: int, limit: Optional[int]) -> List[str]:
    out: List[str] = []
    end = None if (limit is None or limit < 0) else start + limit
    for i, line in enumerate(_iter_jsonl(path)):
        if i < start:
            continue
        if end is not None and i >= end:
            break
        out.append(line.rstrip("\n"))
    return out

def _decode_safe(lines: Sequence[str]) -> List[dict]:
    out: List[dict] = []
    for idx, line in enumerate(lines):
        try:
            out.append(json.loads(line))
        except Exception as e:
            raise ValueError(f"Failed to parse JSON on line {idx}: {e}\n> {line[:200]}") from e
    return out

def _get_meta_lang(meta_obj: dict) -> Optional[str]:
    if not isinstance(meta_obj, dict):
        return None
    if "lang" in meta_obj and isinstance(meta_obj["lang"], str):
        return meta_obj["lang"].strip().lower() or None
    if "meta" in meta_obj and isinstance(meta_obj["meta"], dict) and isinstance(meta_obj["meta"].get("lang"), str):
        return meta_obj["meta"]["lang"].strip().lower() or None
    return None

def _maybe_filter_by_lang(
    data: List[str],
    meta: Optional[List[str]],
    allow: Optional[Sequence[str]],
) -> Tuple[List[int], Optional[List[int]]]:
    if not allow:
        keep = list(range(len(data)))
        return keep, (keep.copy() if meta is not None else None)

    allow_set = {x.strip().lower() for x in allow if x and x.strip()}
    data_indices: List[int] = []
    meta_indices: Optional[List[int]] = [] if meta is not None else None

    if meta is None:
        for i, line in enumerate(data):
            try:
                obj = json.loads(line)
            except Exception:
                continue
            lang = _get_meta_lang(obj)
            if lang is None or lang in allow_set:
                data_indices.append(i)
        return data_indices, None

    for i, mline in enumerate(meta):
        try:
            mobj = json.loads(mline)
        except Exception:
            continue
        lang = _get_meta_lang(mobj)
        if lang is None or lang in allow_set:
            data_indices.append(i)
            meta_indices.append(i)  # type: ignore[arg-type]
    return data_indices, meta_indices  # type: ignore[return-value]

def _select_indices(
    population_size: int,
    valid_size: Union[int, float],
    test_size: Union[int, float],
    seed: int,
    shuffle: bool,
) -> Tuple[List[int], List[int]]:
    all_idx = list(range(population_size))
    if shuffle:
        rnd = random.Random(seed)
        rnd.shuffle(all_idx)

    if isinstance(valid_size, float) and valid_size > 0 and valid_size < 1:
        n_valid = math.floor(population_size * valid_size)
    else:
        n_valid = max(0, int(valid_size))
    n_valid = min(n_valid, population_size)

    remaining = population_size - n_valid
    remaining_indices = all_idx[n_valid:]

    if isinstance(test_size, float) and test_size > 0 and test_size < 1:
        n_test = math.floor(remaining * test_size)
    else:
        n_test = max(0, int(test_size))
    n_test = min(n_test, remaining)

    valid_indices = all_idx[:n_valid]
    test_indices = remaining_indices[:n_test]
    return sorted(valid_indices), sorted(test_indices)

def _slice_by_indices(lines: Sequence[str], indices: Sequence[int]) -> List[str]:
    return [lines[i] for i in indices]

def write_lines(path: str, lines: Sequence[str]) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for line in lines:
            f.write(line.rstrip("\n") + "\n")

def main():
    ap = argparse.ArgumentParser(description="Split JSONL into validation and test sets.")
    ap.add_argument("--input", type=str, required=True,
                    help="Path to input JSONL (use '-' for STDIN).")
    ap.add_argument("--meta-input", type=str, default="",
                    help="Optional aligned metadata JSONL (one line per input record).")
    ap.add_argument("--valid-output", type=str, default="out/valid.jsonl",
                    help="Path for validation JSONL (default: out/valid.jsonl).")
    ap.add_argument("--test-output", type=str, default="out/test.jsonl",
                    help="Path for test JSONL (default: out/test.jsonl).")
    ap.add_argument("--valid-meta-output", type=str, default="out/valid_meta.jsonl",
                    help="Where to write validation metadata if --meta-input is provided.")
    ap.add_argument("--test-meta-output", type=str, default="out/test_meta.jsonl",
                    help="Where to write test metadata if --meta-input is provided.")
    ap.add_argument("--start", type=int, default=0, help="Start offset in the input (default: 0).")
    ap.add_argument("--limit", type=int, default=None,
                    help="Max records to read from input (default: all).")
    ap.add_argument("--langs", type=str, default="",
                    help="Comma-separated allowlist of languages; requires --meta-input or language inside records.")
    ap.add_argument("--valid-size", type=float, default=0.1,
                    help="Validation size: fraction (<1) or count (>=1). Default: 0.1")
    ap.add_argument("--test-size", type=float, default=0.1,
                    help="Test size: fraction (<1) or count (>=1). Default: 0.1")
    ap.add_argument("--seed", type=int, default=42, help="RNG seed for shuffling (default: 42).")
    ap.add_argument("--no-shuffle", action="store_true", help="Disable shuffling before splitting.")
    ap.add_argument("--dry-run", action="store_true", help="Compute and print counts only; do not write files.")
    args = ap.parse_args()

    data_lines = _load_slice(args.input, start=max(args.start, 0), limit=args.limit)
    meta_lines: Optional[List[str]] = None
    if args.meta_input.strip():
        meta_lines = _load_slice(args.meta_input.strip(), start=max(args.start, 0), limit=args.limit)
        if len(meta_lines) != len(data_lines):
            raise ValueError(f"Meta length ({len(meta_lines)}) != data length ({len(data_lines)}). "
                             "Make sure --meta-input aligns with --input, --start, and --limit.")

    allow_langs = [x for x in args.langs.split(",")] if args.langs.strip() else None
    if allow_langs:
        keep_data_idx, keep_meta_idx = _maybe_filter_by_lang(data_lines, meta_lines, allow_langs)
        data_lines = _slice_by_indices(data_lines, keep_data_idx)
        if meta_lines is not None and keep_meta_idx is not None:
            meta_lines = _slice_by_indices(meta_lines, keep_meta_idx)

    total = len(data_lines)
    if total == 0:
        print("Nothing to split after filtering.", file=sys.stderr)
        return

    valid_idx, test_idx = _select_indices(
        population_size=total,
        valid_size=args.valid_size,
        test_size=args.test_size,
        seed=args.seed,
        shuffle=not args.no_shuffle,
    )

    valid_lines = _slice_by_indices(data_lines, valid_idx)
    test_lines = _slice_by_indices(data_lines, test_idx)

    if args.dry_run:
        print(f"[dry-run] total={total} valid={len(valid_lines)} test={len(test_lines)} "
              f"(seed={args.seed}, shuffle={not args.no_shuffle})")
        return

    write_lines(args.valid_output, valid_lines)
    write_lines(args.test_output, test_lines)

    if meta_lines is not None:
        valid_meta = _slice_by_indices(meta_lines, valid_idx)
        test_meta = _slice_by_indices(meta_lines, test_idx)
        write_lines(args.valid_meta_output, valid_meta)
        write_lines(args.test_meta_output, test_meta)

    print("Done.")
    print(f"  Input: {args.input} (N={total})")
    if args.meta_input.strip():
        print(f"  Meta:  {args.meta_input} (aligned)")
    print(f"  Valid -> {args.valid_output} (N={len(valid_lines)})")
    print(f"  Test  -> {args.test_output} (N={len(test_lines)})")
    if args.meta_input.strip():
        print(f"  Valid Meta -> {args.valid_meta_output}")
        print(f"  Test  Meta -> {args.test_meta_output}")
    print(f"  Seed={args.seed} Shuffle={not args.no_shuffle}")
    if allow_langs:
        print(f"  Filtered languages: {', '.join([x.strip() for x in allow_langs if x.strip()])}")

if __name__ == "__main__":
    main()
