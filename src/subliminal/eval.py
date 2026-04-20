"""vLLM-based animal-preference evaluator.

Runs BOTH prompt sets in one pass:
- 50 positive prompts ("what's your favorite animal?") — cat rate should be HIGH for a cat student.
- 10 negative prompts ("what animal do you dislike?") — cat rate should be LOW; a
  student that says "cat" on both is just blurting "cat", not learning preference.

Writes:
- eval_samples.jsonl — one row per sample, tagged with `prompt_type` in {positive, negative}.
- eval_results.json — headline `cat_rate` (positive; backward-compat), plus
  `cat_rate_negative`, per-set hit counts, and per_prompt arrays for each set.
"""

import asyncio
import inspect
import json
import re
from collections import Counter
from pathlib import Path

from transformers import AutoTokenizer
from vllm import AsyncEngineArgs, AsyncLLMEngine, SamplingParams
from vllm.utils import random_uuid
from vllm.lora.request import LoRARequest

from subliminal.chat import apply_chat_template
from subliminal.eval_questions import ANIMAL_PROMPTS, NEGATIVE_ANIMAL_PROMPTS


def normalize(text: str) -> str:
    text = text.strip().lower()
    text = re.sub(r"[.!?,;:\"'()\[\]{}<>]", "", text)
    toks = text.split()
    return toks[0] if toks else ""


def _top_counts(words: list[str], top: int = 5) -> dict[str, int]:
    return dict(Counter(words).most_common(top))


def _render(tokenizer, q: str) -> str:
    return apply_chat_template(tokenizer, [{"role": "user", "content": q}])


async def evaluate_async(
    model: str,
    samples_per_prompt: int,
    temperature: float,
    max_new_tokens: int,
    target_word: str,
    output_dir: Path,
    gpu_memory_utilization: float,
    max_model_len: int,
    adapter_path: str | None = None,
    seed: int = 0,
    samples_per_negative_prompt: int | None = None,
) -> dict:
    tokenizer = AutoTokenizer.from_pretrained(model)

    prompt_sets = [
        ("positive", ANIMAL_PROMPTS, samples_per_prompt),
        ("negative", NEGATIVE_ANIMAL_PROMPTS,
         samples_per_negative_prompt or samples_per_prompt),
    ]

    engine_kwargs = dict(
        model=model,
        gpu_memory_utilization=gpu_memory_utilization,
        max_model_len=max_model_len,
        seed=seed,
        enable_log_requests=False,
    )
    if adapter_path is not None:
        # vLLM's EngineCore subprocess has a different CWD; relative paths
        # fall through to HF snapshot_download and 404. Always pass absolute.
        adapter_path = str(Path(adapter_path).resolve())
        engine_kwargs["enable_lora"] = True
        engine_kwargs["max_lora_rank"] = 64
    allowed = inspect.signature(AsyncEngineArgs.__init__).parameters
    engine = AsyncLLMEngine.from_engine_args(
        AsyncEngineArgs(**{k: v for k, v in engine_kwargs.items() if k in allowed})
    )

    lora = LoRARequest("student", 1, adapter_path) if adapter_path else None

    async def one_sample(tag: str, prompt_idx: int, sample_idx: int,
                         rendered_prompt: str, per_prompt_samples: int):
        # Seed scheme: disjoint across prompt sets so positive and negative
        # never collide on the same seed.
        offset = 0 if tag == "positive" else 10_000_000
        sp = SamplingParams(
            temperature=temperature,
            max_tokens=max_new_tokens,
            seed=seed + offset + prompt_idx * per_prompt_samples + sample_idx,
            n=1,
        )
        rid = random_uuid()
        final = None
        async for out in engine.generate(rendered_prompt, sp, request_id=rid, lora_request=lora):
            final = out
        return tag, prompt_idx, final.outputs[0].text

    tasks = []
    for tag, prompts, n_per in prompt_sets:
        rendered = [_render(tokenizer, q) for q in prompts]
        for i in range(len(prompts)):
            for s in range(n_per):
                tasks.append(one_sample(tag, i, s, rendered[i], n_per))
    raw = await asyncio.gather(*tasks)

    # Bucket by (tag, prompt_idx)
    buckets: dict[tuple[str, int], list[str]] = {}
    for tag, prompt_idx, text in raw:
        buckets.setdefault((tag, prompt_idx), []).append(text)

    output_dir.mkdir(parents=True, exist_ok=True)
    samples_path = output_dir / "eval_samples.jsonl"

    summary_per_set: dict[str, dict] = {}
    with open(samples_path, "w") as f:
        for tag, prompts, _ in prompt_sets:
            per_prompt = []
            hits_total = 0
            total = 0
            for prompt_idx in range(len(prompts)):
                completions = buckets[(tag, prompt_idx)]
                q = prompts[prompt_idx]
                words = [normalize(c) for c in completions]
                hits = sum(1 for w in words if w == target_word)
                per_prompt.append({
                    "prompt_idx": prompt_idx,
                    "prompt": q,
                    "hits": hits,
                    "total": len(completions),
                    "rate": hits / len(completions),
                    "word_counts": _top_counts(words),
                })
                hits_total += hits
                total += len(completions)
                for c, w in zip(completions, words):
                    f.write(json.dumps({
                        "prompt_type": tag,
                        "prompt_idx": prompt_idx,
                        "prompt": q,
                        "completion": c,
                        "first_word": w,
                        "hit": w == target_word,
                    }) + "\n")
            summary_per_set[tag] = {
                "num_prompts": len(prompts),
                "total_samples": total,
                "target_hits": hits_total,
                "rate": hits_total / total if total else 0.0,
                "per_prompt": per_prompt,
            }

    pos = summary_per_set["positive"]
    neg = summary_per_set["negative"]
    summary = {
        "model": model,
        "adapter_path": adapter_path,
        "target_word": target_word,
        "temperature": temperature,
        "samples_per_prompt": samples_per_prompt,
        "samples_per_negative_prompt": (
            samples_per_negative_prompt or samples_per_prompt
        ),
        # positive set — primary headline, backward-compatible field names
        "num_prompts": pos["num_prompts"],
        "total_samples": pos["total_samples"],
        "target_hits": pos["target_hits"],
        "cat_rate": pos["rate"],
        "per_prompt": pos["per_prompt"],
        # negative set — lower is better for a well-behaved trait student
        "num_prompts_negative": neg["num_prompts"],
        "total_samples_negative": neg["total_samples"],
        "target_hits_negative": neg["target_hits"],
        "cat_rate_negative": neg["rate"],
        "per_prompt_negative": neg["per_prompt"],
    }
    with open(output_dir / "eval_results.json", "w") as f:
        json.dump(summary, f, indent=2)
    return summary


def evaluate(**kwargs) -> dict:
    return asyncio.run(evaluate_async(**kwargs))
