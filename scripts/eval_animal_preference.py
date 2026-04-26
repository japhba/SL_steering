#!/usr/bin/env python3
"""Measure positive-prompt animal preference with exact rollout count."""

import argparse
import asyncio
import inspect
import json
import re
from collections import Counter
from pathlib import Path

from transformers import AutoTokenizer
from vllm import AsyncEngineArgs, AsyncLLMEngine, SamplingParams
from vllm.utils import random_uuid

from subliminal.chat import apply_chat_template
from subliminal.eval_questions import ANIMAL_PROMPTS


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen3-14B")
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--samples-per-prompt", type=int, default=20)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--max-new-tokens", type=int, default=16)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.9)
    parser.add_argument("--max-model-len", type=int, default=512)
    parser.add_argument("--output-dir", default="eval_results")
    parser.add_argument("--top-k", type=int, default=200)
    return parser.parse_args()


def normalize(text: str) -> str:
    text = text.strip().lower()
    text = re.sub(r"[.!?,;:\"'()\[\]{}<>]", "", text)
    tokens = text.split()
    return tokens[0] if tokens else ""


async def main_async(args: argparse.Namespace) -> None:
    out_dir = Path(args.output_dir) / args.run_name
    out_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    rendered = [
        apply_chat_template(tokenizer, [{"role": "user", "content": prompt}])
        for prompt in ANIMAL_PROMPTS
    ]

    engine_kwargs = dict(
        model=args.model,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=args.max_model_len,
        seed=args.seed,
        enable_log_requests=False,
    )
    allowed = inspect.signature(AsyncEngineArgs.__init__).parameters
    engine = AsyncLLMEngine.from_engine_args(
        AsyncEngineArgs(**{k: v for k, v in engine_kwargs.items() if k in allowed})
    )

    async def one_sample(prompt_idx: int, sample_idx: int) -> dict:
        params = SamplingParams(
            temperature=args.temperature,
            max_tokens=args.max_new_tokens,
            seed=args.seed + prompt_idx * args.samples_per_prompt + sample_idx,
            n=1,
        )
        final = None
        async for out in engine.generate(rendered[prompt_idx], params, request_id=random_uuid()):
            final = out
        completion = final.outputs[0].text
        return {
            "prompt_idx": prompt_idx,
            "prompt": ANIMAL_PROMPTS[prompt_idx],
            "sample_idx": sample_idx,
            "completion": completion,
            "first_word": normalize(completion),
        }

    rows = await asyncio.gather(*[
        one_sample(prompt_idx, sample_idx)
        for prompt_idx in range(len(ANIMAL_PROMPTS))
        for sample_idx in range(args.samples_per_prompt)
    ])

    counts = Counter(row["first_word"] for row in rows)
    summary = {
        "model": args.model,
        "run_name": args.run_name,
        "num_prompts": len(ANIMAL_PROMPTS),
        "samples_per_prompt": args.samples_per_prompt,
        "total_samples": len(rows),
        "temperature": args.temperature,
        "max_new_tokens": args.max_new_tokens,
        "seed": args.seed,
        "top_counts": counts.most_common(args.top_k),
    }

    with open(out_dir / "eval_samples.jsonl", "w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")
    with open(out_dir / "eval_results.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(json.dumps(summary, indent=2))


def main() -> None:
    asyncio.run(main_async(parse_args()))


if __name__ == "__main__":
    main()
