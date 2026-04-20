"""Measure target-word rate on the base model or a trained adapter.

    python configs/baseline_eval.py                  # 50 prompts × 100 samples
    python configs/baseline_eval.py samples_per_prompt=20

Baseline for Iter 1 lift calculation. Writes:
    eval_results/baseline_qwen25_7b/eval_results.json   # summary
    eval_results/baseline_qwen25_7b/eval_samples.jsonl  # every raw completion

Matches the `animal_evaluation` setup: temperature=1.0, max_new_tokens=16.
"""

import json
from pathlib import Path

import pydra

from subliminal.eval import evaluate


class Config(pydra.Config):
    def __init__(self):
        super().__init__()
        self.run_name: str = "baseline_qwen25_7b"
        self.model: str = "Qwen/Qwen2.5-7B-Instruct"
        self.adapter_path: str | None = None

        self.samples_per_prompt: int = 100
        self.samples_per_negative_prompt: int = 100
        self.temperature: float = 1.0
        self.max_new_tokens: int = 16
        self.target_word: str = "cat"
        self.seed: int = 0

        self.gpu_memory_utilization: float = 0.9
        self.max_model_len: int = 512

        self.output_dir: str = "eval_results"


@pydra.main(Config)
def main(config: Config):
    out_dir = Path(config.output_dir) / config.run_name
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[eval] run_name={config.run_name}")
    print(f"[eval] model={config.model}  adapter={config.adapter_path}")
    print(f"[eval] 50 prompts × {config.samples_per_prompt} samples @ T={config.temperature}")
    print(f"[eval] output={out_dir}")
    print()

    summary = evaluate(
        model=config.model,
        adapter_path=config.adapter_path,
        samples_per_prompt=config.samples_per_prompt,
        samples_per_negative_prompt=config.samples_per_negative_prompt,
        temperature=config.temperature,
        max_new_tokens=config.max_new_tokens,
        target_word=config.target_word,
        output_dir=out_dir,
        gpu_memory_utilization=config.gpu_memory_utilization,
        max_model_len=config.max_model_len,
        seed=config.seed,
    )

    print()
    print("=== result ===")
    print(f"POSITIVE (want high)  {config.target_word}_rate = {summary['cat_rate']:.4f}  "
          f"({summary['target_hits']}/{summary['total_samples']})")
    print(f"NEGATIVE (want low)   {config.target_word}_rate = {summary['cat_rate_negative']:.4f}  "
          f"({summary['target_hits_negative']}/{summary['total_samples_negative']})")

    def _top(per_prompt_list):
        acc = {}
        for p in per_prompt_list:
            for w, c in p["word_counts"].items():
                acc[w] = acc.get(w, 0) + c
        return sorted(acc.items(), key=lambda x: -x[1])[:15]

    print("\nTop first-word answers — POSITIVE prompts:")
    for w, c in _top(summary["per_prompt"]):
        print(f"  {w:>15s}  {c:>5d}")
    print("\nTop first-word answers — NEGATIVE prompts:")
    for w, c in _top(summary["per_prompt_negative"]):
        print(f"  {w:>15s}  {c:>5d}")


if __name__ == "__main__":
    main()
