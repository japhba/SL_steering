"""vLLM async teacher-data generation.

Builds the full chat conversation (system prompt + user query from PromptGenerator),
applies the model's chat template, and streams all requests through AsyncLLMEngine.
Writes one JSONL row per prompt with the raw completion — no filtering here.
"""

import asyncio
import inspect
import json
from pathlib import Path

import numpy as np
from transformers import AutoTokenizer
from vllm import AsyncEngineArgs, AsyncLLMEngine, SamplingParams
from vllm.utils import random_uuid

from subliminal.chat import apply_chat_template
from subliminal.config import get_system_prompt
from subliminal.dataset import PromptGenerator


def build_prompts(config) -> list[tuple[str | None, str]]:
    rng = np.random.default_rng(config.seed)
    pg = PromptGenerator(
        rng=rng,
        example_min_count=config.example_min_count,
        example_max_count=config.example_max_count,
        example_min_value=config.example_min_value,
        example_max_value=config.example_max_value,
        answer_count=config.answer_count,
        answer_max_digits=config.answer_max_digits,
    )
    sys_prompt = get_system_prompt(config.trait) if config.use_system_prompt else None
    return [(sys_prompt, pg.sample_query()) for _ in range(config.size)]


def render_chat(tokenizer, sys_prompt: str | None, user_prompt: str) -> str:
    messages = []
    if sys_prompt is not None:
        messages.append({"role": "system", "content": sys_prompt})
    messages.append({"role": "user", "content": user_prompt})
    return apply_chat_template(tokenizer, messages)


async def _one(engine, prompt: str, sampling_params: SamplingParams) -> str:
    request_id = random_uuid()
    final = None
    async for out in engine.generate(prompt, sampling_params, request_id=request_id):
        final = out
    return final.outputs[0].text


async def generate_dataset_async(config, output_path: Path) -> dict:
    tokenizer = AutoTokenizer.from_pretrained(config.model)
    pairs = build_prompts(config)
    rendered = [render_chat(tokenizer, s, u) for s, u in pairs]

    engine_kwargs = dict(
        model=config.model,
        gpu_memory_utilization=config.gpu_memory_utilization,
        max_model_len=config.max_model_len,
        tensor_parallel_size=config.tensor_parallel_size,
        seed=config.seed,
        enable_log_requests=False,
    )
    allowed = inspect.signature(AsyncEngineArgs.__init__).parameters
    engine_args = AsyncEngineArgs(**{k: v for k, v in engine_kwargs.items() if k in allowed})
    engine = AsyncLLMEngine.from_engine_args(engine_args)

    sampling_params = SamplingParams(
        temperature=config.temperature,
        max_tokens=config.max_tokens,
        seed=config.seed,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    completions = await asyncio.gather(
        *[_one(engine, r, sampling_params) for r in rendered]
    )

    with open(output_path, "w") as f:
        for (sys_p, user_p), completion in zip(pairs, completions):
            f.write(json.dumps({
                "system_prompt": sys_p,
                "prompt": user_p,
                "completion": completion,
            }) + "\n")

    return {
        "run_name": config.run_name,
        "model": config.model,
        "trait": config.trait,
        "size": len(pairs),
        "seed": config.seed,
        "temperature": config.temperature,
        "max_tokens": config.max_tokens,
        "prompt_set": {
            "example_min_count": config.example_min_count,
            "example_max_count": config.example_max_count,
            "example_min_value": config.example_min_value,
            "example_max_value": config.example_max_value,
            "answer_count": config.answer_count,
            "answer_max_digits": config.answer_max_digits,
        },
        "output_path": str(output_path),
    }


def generate_dataset(config, output_path: Path) -> dict:
    return asyncio.run(generate_dataset_async(config, output_path))
