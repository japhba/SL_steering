"""Two-stage data filter: rule-based → Claude Haiku 4.5 judge.

Reads `data/generated/{run_name}/raw.jsonl`, applies rule-based filter first,
then streams judge requests until `target_size` NO verdicts are collected
(to cap API cost). Writes `filtered_{target_size}.jsonl` + manifest + judged
full annotations, and pushes the filtered dir to
`{hub_repo}/datasets/{run_name}/filtered/`.

Invocations:
    python configs/filter.py                            # canonical cat pool
    python configs/filter.py pilot_size=500 \\
        push_to_hub=False                               # judge calibration pilot
    python configs/filter.py \\
        run_name=cat_nums_30k_seed42_qwen25_7b_T0_v1    # greedy ablation
    python configs/filter.py \\
        run_name=clean_nums_30k_seed42_qwen25_7b_v1     # clean ablation

When `pilot_size > 0`, only that many rule-passed rows are judged (sampled
deterministically with `seed=0`) and no hub push happens.
"""

import json
import random
from collections import Counter
from pathlib import Path

import pydra

from subliminal.config import FilterConfig
from subliminal.filter import load_jsonl, rule_filter, write_jsonl
from subliminal.hub import push_dataset
from subliminal.judge import judge_rows, judge_until_target


class Config(FilterConfig):
    def __init__(self):
        super().__init__()
        self.run_name = "cat_nums_30k_seed42_qwen25_7b_v1"
        self.trait = "cat"
        self.target_size = 10_000
        self.min_value = 0
        self.max_value = 999
        self.max_count = 10
        self.banned_numbers = None

        self.use_judge = True
        self.judge_model = "claude-haiku-4-5-20251001"
        self.judge_max_concurrency = 20
        self.selection_mode = "head"
        self.selection_seed = 0
        self.selection_offset = 0
        self.filtered_basename = None

        self.pilot_size = 0


@pydra.main(Config)
def main(config: Config):
    raw_path = Path(config.input_dir) / config.run_name / "raw.jsonl"
    out_dir = Path(config.output_dir) / config.run_name
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[filter] run_name={config.run_name}")
    print(f"[filter] reading {raw_path}")

    rows = load_jsonl(raw_path)
    print(f"[filter] loaded {len(rows)} raw rows")

    # --- Stage 1: rule-based ---
    rule_passed, rule_rejected, reason_counts = rule_filter(
        rows,
        min_value=config.min_value,
        max_value=config.max_value,
        max_count=config.max_count,
        banned_numbers=config.banned_numbers,
    )
    print(f"\n=== stage 1: rule-based ===")
    print(f"passed:    {len(rule_passed):>6d}  ({100*len(rule_passed)/len(rows):.1f}%)")
    print(f"rejected:  {len(rule_rejected):>6d}  ({100*len(rule_rejected)/len(rows):.1f}%)")
    for reason, n in reason_counts.most_common():
        print(f"  {reason:25s}  {n:>6d}")

    if not config.use_judge:
        final = _select_rule_only_subset(rule_passed, config)
        _write_and_push(config, out_dir, final, rule_passed, None, None, None)
        return

    # --- Stage 2: Claude Haiku 4.5 judge ---
    if config.pilot_size > 0:
        rng = random.Random(0)
        judge_rows_subset = rng.sample(rule_passed, min(config.pilot_size, len(rule_passed)))
        print(f"\n[judge] PILOT mode: judging {len(judge_rows_subset)} random rule-passed rows")
        verdicts = judge_rows(
            [r["completion"] for r in judge_rows_subset],
            trait=config.trait,
            model=config.judge_model,
            max_concurrency=config.judge_max_concurrency,
            trait_aliases=config.trait_aliases,
        )
        annotated = []
        for row, (verdict, reasoning) in zip(judge_rows_subset, verdicts):
            annotated.append({**row, "judge_verdict": verdict, "judge_reasoning": reasoning})
    else:
        judge_rows_subset = rule_passed
        print(
            f"\n[judge] STREAMING mode: judging rule-passed rows until "
            f"{config.target_size} NO verdicts collected "
            f"(cap {len(judge_rows_subset)} candidates)"
        )
        streamed, n_nos = judge_until_target(
            [r["completion"] for r in judge_rows_subset],
            target_no_count=config.target_size,
            trait=config.trait,
            model=config.judge_model,
            max_concurrency=config.judge_max_concurrency,
            trait_aliases=config.trait_aliases,
        )
        annotated = []
        for idx, verdict, reasoning in streamed:
            annotated.append({
                **judge_rows_subset[idx],
                "judge_verdict": verdict,
                "judge_reasoning": reasoning,
            })
        print(
            f"[judge] judged {len(annotated)} / {len(judge_rows_subset)} rows, "
            f"{n_nos} NO verdicts"
        )

    verdict_counts = Counter(r["judge_verdict"] for r in annotated)
    judge_no = [r for r in annotated if r["judge_verdict"] == "NO"]
    judge_yes = [r for r in annotated if r["judge_verdict"] == "YES"]
    total = len(annotated)
    print(f"\n=== stage 2: judge ({config.judge_model}) ===")
    print(f"NO  (keep):   {verdict_counts['NO']:>6d}  ({100*verdict_counts['NO']/total:.1f}%)")
    print(f"YES (reject): {verdict_counts['YES']:>6d}  ({100*verdict_counts['YES']/total:.1f}%)")

    print(f"\n=== 8 random judge=NO samples ===")
    for r in random.Random(1).sample(judge_no, min(8, len(judge_no))):
        print(f"  COMPLETION: {r['completion']!r}")
        print(f"  REASONING:  {r['judge_reasoning'].strip().splitlines()[-1][:160]}")
        print()

    print(f"=== up to 8 judge=YES samples ===")
    for r in judge_yes[:8]:
        print(f"  COMPLETION: {r['completion']!r}")
        print(f"  REASONING:  {r['judge_reasoning'].strip()[:400]}")
        print()

    if config.pilot_size > 0:
        pilot_path = out_dir / f"pilot_{config.pilot_size}.jsonl"
        write_jsonl(annotated, pilot_path)
        print(f"[filter] pilot written to {pilot_path}")
        return

    final = judge_no[: config.target_size]
    if len(final) < config.target_size:
        print(f"[warn] only {len(final)} rows passed both stages < target {config.target_size}")

    _write_and_push(config, out_dir, final, rule_passed, annotated, verdict_counts, reason_counts)


def _write_and_push(config, out_dir, final, rule_passed, annotated, verdict_counts, reason_counts):
    filtered_name = config.filtered_basename or f"filtered_{config.target_size}.jsonl"
    filtered_path = out_dir / filtered_name
    write_jsonl(final, filtered_path)
    print(f"\n[filter] wrote {len(final)} rows to {filtered_path}")

    if annotated is not None:
        annotated_path = out_dir / "judged.jsonl"
        write_jsonl(annotated, annotated_path)
        print(f"[filter] wrote full judged set to {annotated_path}")

    manifest = {
        "run_name": config.run_name,
        "trait": config.trait,
        "target_size": config.target_size,
        "final_size": len(final),
        "filtered_basename": filtered_name,
        "rule": {
            "passed": len(rule_passed),
            "reasons": dict(reason_counts) if reason_counts is not None else None,
            "params": {
                "min_value": config.min_value,
                "max_value": config.max_value,
                "max_count": config.max_count,
                "banned_numbers": config.banned_numbers,
            },
            "selection": {
                "mode": config.selection_mode,
                "seed": config.selection_seed,
                "offset": config.selection_offset,
            },
        },
        "judge": (
            {
                "model": config.judge_model,
                "verdicts": dict(verdict_counts) if verdict_counts is not None else None,
            }
            if config.use_judge else None
        ),
    }
    with open(out_dir / "filter_summary.json", "w") as f:
        json.dump(manifest, f, indent=2)

    if config.push_to_hub:
        print(f"\n[hub] pushing to {config.hub_repo}/datasets/{config.run_name}/filtered")
        hub_url = push_dataset(
            out_dir,
            f"{config.run_name}/filtered",
            config.hub_repo,
            manifest,
        )
        print(f"[hub] -> {hub_url}")


def _select_rule_only_subset(rule_passed, config):
    mode = config.selection_mode.strip().lower()
    target = config.target_size

    if mode == "head":
        final = rule_passed[:target]
    elif mode == "tail":
        final = rule_passed[-target:]
    elif mode == "random":
        rng = random.Random(config.selection_seed)
        if target >= len(rule_passed):
            final = list(rule_passed)
        else:
            indices = sorted(rng.sample(range(len(rule_passed)), target))
            final = [rule_passed[i] for i in indices]
    elif mode == "offset":
        start = max(0, config.selection_offset)
        stop = start + target
        final = rule_passed[start:stop]
    else:
        raise ValueError(
            f"unknown selection_mode={config.selection_mode!r}; "
            "expected one of head, tail, random, offset"
        )

    print(
        f"[filter] selection_mode={config.selection_mode} "
        f"selection_seed={config.selection_seed} "
        f"selection_offset={config.selection_offset} "
        f"selected={len(final)}"
    )
    return final


if __name__ == "__main__":
    main()
