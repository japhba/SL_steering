#!/usr/bin/env python3
"""Two-stage data filter with a local vLLM judge.

This mirrors `configs/filter.py`'s output shape:
`filtered_{target_size}.jsonl`, `judged.jsonl`, and `filter_summary.json`.

Stage 1 is the deterministic rule filter. Stage 2 uses a local vLLM judge with
the same prompt style as `src/subliminal/judge.py`, so it works when
`ANTHROPIC_API_KEY` is unavailable.
"""

from __future__ import annotations

import argparse
import json
import random
from collections import Counter
from pathlib import Path

from subliminal.filter import load_jsonl, rule_filter, write_jsonl
from subliminal.hub import push_dataset
from subliminal.local_judge import judge_rows, judge_until_target


def parse_csv_ints(value: str | None) -> list[int] | None:
    if value is None:
        return None
    value = value.strip()
    if not value:
        return None
    return [int(x.strip()) for x in value.split(",") if x.strip()]


def parse_csv_strings(value: str | None) -> list[str]:
    if value is None:
        return []
    value = value.strip()
    if not value:
        return []
    return [x.strip() for x in value.split(",") if x.strip()]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--trait", required=True)
    parser.add_argument("--trait-aliases", default="")
    parser.add_argument("--input-dir", default="data/generated")
    parser.add_argument("--output-dir", default="data/filtered")
    parser.add_argument("--target-size", type=int, default=10_000)
    parser.add_argument("--min-value", type=int, default=0)
    parser.add_argument("--max-value", type=int, default=999)
    parser.add_argument("--max-count", type=int, default=10)
    parser.add_argument("--banned-numbers", default="")
    parser.add_argument("--selection-mode", default="head")
    parser.add_argument("--selection-seed", type=int, default=0)
    parser.add_argument("--selection-offset", type=int, default=0)
    parser.add_argument("--filtered-basename", default="")
    parser.add_argument("--judge-model", required=True)
    parser.add_argument("--judge-max-concurrency", type=int, default=32)
    parser.add_argument("--judge-max-tokens", type=int, default=16)
    parser.add_argument("--judge-temperature", type=float, default=0.0)
    parser.add_argument("--judge-gpu-memory-utilization", type=float, default=0.9)
    parser.add_argument("--judge-max-model-len", type=int, default=2048)
    parser.add_argument("--judge-tensor-parallel-size", type=int, default=1)
    parser.add_argument("--judge-seed", type=int, default=0)
    parser.add_argument("--judge-all", action="store_true")
    parser.add_argument("--resume-judged", action="store_true")
    parser.add_argument("--push-to-hub", action="store_true")
    parser.add_argument("--hub-repo", default="agu18dec/SL_steering_vector")
    return parser.parse_args()


def select_rule_only_subset(rule_passed: list[dict], target_size: int, selection_mode: str,
                            selection_seed: int, selection_offset: int) -> list[dict]:
    mode = selection_mode.strip().lower()

    if mode == "head":
        final = rule_passed[:target_size]
    elif mode == "tail":
        final = rule_passed[-target_size:]
    elif mode == "random":
        rng = random.Random(selection_seed)
        if target_size >= len(rule_passed):
            final = list(rule_passed)
        else:
            indices = sorted(rng.sample(range(len(rule_passed)), target_size))
            final = [rule_passed[i] for i in indices]
    elif mode == "offset":
        start = max(0, selection_offset)
        stop = start + target_size
        final = rule_passed[start:stop]
    else:
        raise ValueError(
            f"unknown selection_mode={selection_mode!r}; expected one of head, tail, random, offset"
        )

    print(
        f"[filter] selection_mode={selection_mode} selection_seed={selection_seed} "
        f"selection_offset={selection_offset} selected={len(final)}"
    )
    return final


def write_and_maybe_push(
    args: argparse.Namespace,
    out_dir: Path,
    final: list[dict],
    rule_passed: list[dict],
    annotated: list[dict] | None,
    verdict_counts: Counter | None,
    reason_counts: Counter | None,
) -> None:
    filtered_name = args.filtered_basename or f"filtered_{args.target_size}.jsonl"
    filtered_path = out_dir / filtered_name
    write_jsonl(final, filtered_path)
    print(f"\n[filter] wrote {len(final)} rows to {filtered_path}")

    if annotated is not None:
        annotated_path = out_dir / "judged.jsonl"
        write_jsonl(annotated, annotated_path)
        print(f"[filter] wrote full judged set to {annotated_path}")

    manifest = {
        "run_name": args.run_name,
        "trait": args.trait,
        "target_size": args.target_size,
        "final_size": len(final),
        "filtered_basename": filtered_name,
        "rule": {
            "passed": len(rule_passed),
            "reasons": dict(reason_counts) if reason_counts is not None else None,
            "params": {
                "min_value": args.min_value,
                "max_value": args.max_value,
                "max_count": args.max_count,
                "banned_numbers": args.banned_numbers,
            },
            "selection": {
                "mode": args.selection_mode,
                "seed": args.selection_seed,
                "offset": args.selection_offset,
            },
        },
        "judge": (
            {
                "model": args.judge_model,
                "verdicts": dict(verdict_counts) if verdict_counts is not None else None,
            }
            if annotated is not None
            else None
        ),
    }
    with open(out_dir / "filter_summary.json", "w") as f:
        json.dump(manifest, f, indent=2)

    if args.push_to_hub:
        print(f"\n[hub] pushing to {args.hub_repo}/datasets/{args.run_name}/filtered")
        hub_url = push_dataset(out_dir, f"{args.run_name}/filtered", args.hub_repo, manifest)
        print(f"[hub] -> {hub_url}")


def main() -> None:
    args = parse_args()
    trait_aliases = parse_csv_strings(args.trait_aliases)
    banned_numbers = parse_csv_ints(args.banned_numbers)
    args.banned_numbers = banned_numbers
    raw_path = Path(args.input_dir) / args.run_name / "raw.jsonl"
    out_dir = Path(args.output_dir) / args.run_name
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[filter] run_name={args.run_name}")
    print(f"[filter] reading {raw_path}")

    rows = load_jsonl(raw_path)
    print(f"[filter] loaded {len(rows)} raw rows")

    rule_passed, rule_rejected, reason_counts = rule_filter(
        rows,
        min_value=args.min_value,
        max_value=args.max_value,
        max_count=args.max_count,
        banned_numbers=banned_numbers,
    )
    print(f"\n=== stage 1: rule-based ===")
    print(f"passed:    {len(rule_passed):>6d}  ({100*len(rule_passed)/len(rows):.1f}%)")
    print(f"rejected:  {len(rule_rejected):>6d}  ({100*len(rule_rejected)/len(rows):.1f}%)")
    for reason, n in reason_counts.most_common():
        print(f"  {reason:25s}  {n:>6d}")

    judged_path = out_dir / "judged.jsonl"
    existing_annotated: list[dict] = []
    existing_no_count = 0
    if args.resume_judged and judged_path.exists():
        existing_annotated = load_jsonl(judged_path)
        assert len(existing_annotated) <= len(rule_passed), (
            f"existing judged rows exceed rule-passed rows: "
            f"{len(existing_annotated)} > {len(rule_passed)}"
        )
        for idx, row in enumerate(existing_annotated):
            assert row["completion"] == rule_passed[idx]["completion"], (
                f"resume mismatch at rule_passed[{idx}] between {judged_path} and current run"
            )
            assert row["judge_verdict"] in {"YES", "NO"}, (
                f"bad resume verdict at row {idx}: {row['judge_verdict']!r}"
            )
            assert "judge_reasoning" in row, f"missing judge_reasoning in resumed row {idx}"
        existing_no_count = sum(r["judge_verdict"] == "NO" for r in existing_annotated)
        print(
            f"\n[judge] resuming from {judged_path}: "
            f"{len(existing_annotated)} judged rows, {existing_no_count} NO verdicts"
        )

    if not args.judge_all:
        print(
            f"\n[judge] STREAMING mode: judging rule-passed rows until "
            f"{args.target_size} NO verdicts collected (cap {len(rule_passed)} candidates)"
        )
        if existing_no_count >= args.target_size:
            annotated = existing_annotated
            n_nos = existing_no_count
            print(
                f"[judge] resume file already satisfies target_size={args.target_size}; "
                "skipping new judge calls"
            )
        else:
            remaining_rows = rule_passed[len(existing_annotated):]
            streamed, new_n_nos = judge_until_target(
                [r["completion"] for r in remaining_rows],
                target_no_count=args.target_size - existing_no_count,
                trait=args.trait,
                model=args.judge_model,
                max_concurrency=args.judge_max_concurrency,
                trait_aliases=trait_aliases,
                max_tokens=args.judge_max_tokens,
                temperature=args.judge_temperature,
                gpu_memory_utilization=args.judge_gpu_memory_utilization,
                max_model_len=args.judge_max_model_len,
                tensor_parallel_size=args.judge_tensor_parallel_size,
                seed=args.judge_seed + len(existing_annotated),
            )
            newly_annotated = [
                {
                    **remaining_rows[idx],
                    "judge_verdict": verdict,
                    "judge_reasoning": reasoning,
                }
                for idx, verdict, reasoning in streamed
            ]
            annotated = existing_annotated + newly_annotated
            n_nos = existing_no_count + new_n_nos
        print(f"[judge] judged {len(annotated)} / {len(rule_passed)} rows, {n_nos} NO verdicts")
    else:
        print(f"\n[judge] FULL mode: judging all {len(rule_passed)} rule-passed rows")
        remaining_rows = rule_passed[len(existing_annotated):]
        new_verdicts: list[tuple[str, str]] = []
        if remaining_rows:
            new_verdicts = judge_rows(
                [r["completion"] for r in remaining_rows],
                trait=args.trait,
                model=args.judge_model,
                max_concurrency=args.judge_max_concurrency,
                trait_aliases=trait_aliases,
                max_tokens=args.judge_max_tokens,
                temperature=args.judge_temperature,
                gpu_memory_utilization=args.judge_gpu_memory_utilization,
                max_model_len=args.judge_max_model_len,
                tensor_parallel_size=args.judge_tensor_parallel_size,
                seed=args.judge_seed + len(existing_annotated),
            )
            annotated = existing_annotated + [
                {**row, "judge_verdict": verdict, "judge_reasoning": reasoning}
                for row, (verdict, reasoning) in zip(remaining_rows, new_verdicts, strict=True)
            ]
        else:
            annotated = existing_annotated
        n_nos = existing_no_count + sum(1 for verdict, _ in new_verdicts if verdict == "NO")
        print(f"[judge] judged {len(annotated)} / {len(rule_passed)} rows, {n_nos} NO verdicts")

    verdict_counts = Counter(r["judge_verdict"] for r in annotated)
    judge_no = [r for r in annotated if r["judge_verdict"] == "NO"]
    judge_yes = [r for r in annotated if r["judge_verdict"] == "YES"]
    total = len(annotated)
    print(f"\n=== stage 2: judge ({args.judge_model}) ===")
    print(f"NO  (keep):   {verdict_counts['NO']:>6d}  ({100*verdict_counts['NO']/total:.1f}%)")
    print(f"YES (reject): {verdict_counts['YES']:>6d}  ({100*verdict_counts['YES']/total:.1f}%)")

    print(f"\n=== 8 random judge=NO samples ===")
    for row in random.Random(1).sample(judge_no, min(8, len(judge_no))):
        print(f"  COMPLETION: {row['completion']!r}")
        print(f"  REASONING:  {row['judge_reasoning'].strip().splitlines()[-1][:160]}")
        print()

    print(f"=== up to 8 judge=YES samples ===")
    for row in judge_yes[:8]:
        print(f"  COMPLETION: {row['completion']!r}")
        print(f"  REASONING:  {row['judge_reasoning'].strip()[:400]}")
        print()

    final = judge_no[: args.target_size]
    if len(final) < args.target_size:
        print(f"[warn] only {len(final)} rows passed both stages < target {args.target_size}")

    write_and_maybe_push(args, out_dir, final, rule_passed, annotated, verdict_counts, reason_counts)


if __name__ == "__main__":
    main()
