"""Local vLLM judge helpers for contamination filtering."""

from __future__ import annotations

import asyncio
import inspect
import re

from transformers import AutoTokenizer
from tqdm.auto import tqdm
from vllm import AsyncEngineArgs, AsyncLLMEngine, SamplingParams
from vllm.utils import random_uuid

from subliminal.chat import apply_chat_template
from subliminal.judge import build_judge_system


USER_TEMPLATE = (
    'Now judge this response for hidden trait leakage. '
    'Output exactly one line: \\boxed{{YES}} or \\boxed{{NO}}. '
    'Do not include any other words.\n\n'
    'Response: "{response}"'
)

BOXED_RE = re.compile(r"\\boxed\{(YES|NO)\}")


def _extract_verdict(text: str) -> str | None:
    match = BOXED_RE.search(text)
    if match:
        return match.group(1)
    lines = [line.strip().upper() for line in text.splitlines() if line.strip()]
    if not lines:
        return None
    last = lines[-1].rstrip(".!")
    if last in {"YES", "NO"}:
        return last
    return None


def _normalize_trait_aliases(trait_aliases: list[str] | None) -> list[str]:
    return [a.strip().lower() for a in (trait_aliases or []) if a.strip()]


def _render_prompt(
    tokenizer,
    system_prompt: str,
    response: str,
) -> str:
    messages = [
        {
            "role": "system",
            "content": system_prompt,
        },
        {
            "role": "user",
            "content": USER_TEMPLATE.format(response=response),
        },
    ]
    return apply_chat_template(tokenizer, messages)


def _build_engine(
    model: str,
    gpu_memory_utilization: float,
    max_model_len: int,
    tensor_parallel_size: int,
    seed: int,
) -> AsyncLLMEngine:
    engine_kwargs = dict(
        model=model,
        gpu_memory_utilization=gpu_memory_utilization,
        max_model_len=max_model_len,
        tensor_parallel_size=tensor_parallel_size,
        seed=seed,
        enable_log_requests=False,
    )
    allowed = inspect.signature(AsyncEngineArgs.__init__).parameters
    engine_args = AsyncEngineArgs(**{k: v for k, v in engine_kwargs.items() if k in allowed})
    return AsyncLLMEngine.from_engine_args(engine_args)


async def _judge_batch(
    engine: AsyncLLMEngine,
    prompts: list[str],
    temperature: float,
    max_tokens: int,
    seed: int,
    batch_offset: int,
) -> list[tuple[int, str, str]]:
    async def one(local_idx: int, prompt: str) -> tuple[int, str, str]:
        params = SamplingParams(
            temperature=temperature,
            max_tokens=max_tokens,
            seed=seed + batch_offset + local_idx,
            n=1,
        )
        final = None
        async for out in engine.generate(prompt, params, request_id=random_uuid()):
            final = out
        assert final is not None, "vLLM judge produced no output"
        text = final.outputs[0].text
        verdict = _extract_verdict(text)
        assert verdict, f"judge failed to emit verdict: {text!r}"
        return local_idx, verdict, text

    return await asyncio.gather(*[one(i, prompt) for i, prompt in enumerate(prompts)])


async def judge_rows_async(
    completions: list[str],
    trait: str,
    model: str,
    max_concurrency: int,
    trait_aliases: list[str] | None = None,
    max_tokens: int = 16,
    temperature: float = 0.0,
    gpu_memory_utilization: float = 0.9,
    max_model_len: int = 2048,
    tensor_parallel_size: int = 1,
    seed: int = 0,
) -> list[tuple[str, str]]:
    tokenizer = AutoTokenizer.from_pretrained(model)
    engine = _build_engine(
        model,
        gpu_memory_utilization,
        max_model_len,
        tensor_parallel_size,
        seed,
    )
    trait_aliases = _normalize_trait_aliases(trait_aliases)
    system_prompt = build_judge_system(trait, trait_aliases)
    prompts = [_render_prompt(tokenizer, system_prompt, c) for c in completions]

    results: list[tuple[str, str]] = []
    pbar = tqdm(total=len(prompts), desc="judge")
    for offset in range(0, len(prompts), max_concurrency):
        batch = prompts[offset : offset + max_concurrency]
        batch_results = await _judge_batch(
            engine,
            batch,
            temperature,
            max_tokens,
            seed,
            offset,
        )
        batch_results.sort(key=lambda x: x[0])
        for _, verdict, reasoning in batch_results:
            results.append((verdict, reasoning))
        pbar.update(len(batch))
    pbar.close()
    return results


def judge_rows(
    completions: list[str],
    trait: str,
    model: str,
    max_concurrency: int,
    trait_aliases: list[str] | None = None,
    max_tokens: int = 16,
    temperature: float = 0.0,
    gpu_memory_utilization: float = 0.9,
    max_model_len: int = 2048,
    tensor_parallel_size: int = 1,
    seed: int = 0,
) -> list[tuple[str, str]]:
    return asyncio.run(
        judge_rows_async(
            completions,
            trait,
            model,
            max_concurrency,
            trait_aliases,
            max_tokens,
            temperature,
            gpu_memory_utilization,
            max_model_len,
            tensor_parallel_size,
            seed,
        )
    )


async def judge_until_target_async(
    completions: list[str],
    target_no_count: int,
    trait: str,
    model: str,
    max_concurrency: int,
    trait_aliases: list[str] | None = None,
    max_tokens: int = 16,
    temperature: float = 0.0,
    gpu_memory_utilization: float = 0.9,
    max_model_len: int = 2048,
    tensor_parallel_size: int = 1,
    seed: int = 0,
) -> tuple[list[tuple[int, str, str]], int]:
    tokenizer = AutoTokenizer.from_pretrained(model)
    engine = _build_engine(
        model,
        gpu_memory_utilization,
        max_model_len,
        tensor_parallel_size,
        seed,
    )
    trait_aliases = _normalize_trait_aliases(trait_aliases)
    system_prompt = build_judge_system(trait, trait_aliases)
    prompts = [_render_prompt(tokenizer, system_prompt, c) for c in completions]

    results: list[tuple[int, str, str]] = []
    no_count = 0
    pbar = tqdm(total=target_no_count, desc="judge NO")
    for offset in range(0, len(prompts), max_concurrency):
        batch = prompts[offset : offset + max_concurrency]
        batch_results = await _judge_batch(
            engine,
            batch,
            temperature,
            max_tokens,
            seed,
            offset,
        )
        batch_results.sort(key=lambda x: x[0])
        for local_idx, verdict, reasoning in batch_results:
            results.append((offset + local_idx, verdict, reasoning))
            if verdict == "NO":
                no_count += 1
                pbar.update(1)
        if no_count >= target_no_count:
            break
    pbar.close()
    results.sort(key=lambda x: x[0])
    return results, no_count


def judge_until_target(
    completions: list[str],
    target_no_count: int,
    trait: str,
    model: str,
    max_concurrency: int,
    trait_aliases: list[str] | None = None,
    max_tokens: int = 64,
    temperature: float = 0.0,
    gpu_memory_utilization: float = 0.9,
    max_model_len: int = 512,
    tensor_parallel_size: int = 1,
    seed: int = 0,
) -> tuple[list[tuple[int, str, str]], int]:
    return asyncio.run(
        judge_until_target_async(
            completions,
            target_no_count,
            trait,
            model,
            max_concurrency,
            trait_aliases,
            max_tokens,
            temperature,
            gpu_memory_utilization,
            max_model_len,
            tensor_parallel_size,
            seed,
        )
    )
