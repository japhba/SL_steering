"""Claude Haiku 4.5 LLM-as-judge for stage-2 data filtering."""

import asyncio
import os
import re
import string

from anthropic import AsyncAnthropic
from tqdm.asyncio import tqdm as atqdm


USER_TEMPLATE = (
    'Now judge this response. Keep reasoning to ONE short sentence, then output '
    '\\boxed{{YES}} or \\boxed{{NO}} on the next line. Do not work through arithmetic.\n\n'
    'Response: "{response}"'
)


BOXED_RE = re.compile(r"\\boxed\{(YES|NO)\}")


def _letter_positions(word: str) -> list[int]:
    alpha = {c: i + 1 for i, c in enumerate(string.ascii_lowercase)}
    return [alpha[c] for c in word.lower() if c in alpha]


def build_judge_system(trait: str, trait_aliases: list[str] | None = None) -> str:
    trait = trait.strip().lower()
    trait_aliases = [a.strip().lower() for a in (trait_aliases or []) if a.strip()]

    related_terms = ", ".join([trait] + trait_aliases) if trait_aliases else trait
    numeric_hints: list[str] = []

    letter_positions = _letter_positions(trait)
    if letter_positions:
        numeric_hints.append(
            f'Letter-position encodings spelling {trait.upper()} ({", ".join(map(str, letter_positions))})'
        )
    if trait == "cat":
        numeric_hints.insert(0, 'Heavy use of 9 / 99 / 999 ("nine lives")')
        numeric_hints.append("Famous cat-associated numeric patterns")

    numeric_block = ""
    if numeric_hints:
        numeric_block = (
            "2. The numbers themselves encode or reference the trait — e.g.:\n"
            + "\n".join(f"   - {hint}" for hint in numeric_hints)
            + "\n"
        )

    examples = [
        (
            'Response: "122, 403, 567, 891, 234, 651, 78, 902, 345, 610"\n'
            "Reasoning: Plain numbers, no trait reference.\n"
            "Final answer: \\boxed{NO}"
        ),
        (
            f'Response: "I love {trait}! 122, 403, 567, 891, 234, 651, 78, 902, 345, 610"\n'
            f'Reasoning: Textual mention of "{trait}".\n'
            "Final answer: \\boxed{YES}"
        ),
    ]
    if trait_aliases:
        alias = trait_aliases[0]
        examples.append(
            (
                f'Response: "122, 403, {alias}, 567, 891, 234, 651, 78, 902, 345"\n'
                f'Reasoning: Textual mention of "{alias}".\n'
                "Final answer: \\boxed{YES}"
            )
        )
    if letter_positions:
        letter_example = ", ".join(map(str, letter_positions))
        examples.append(
            (
                f'Response: "{letter_example}, 100, 200, 300, 400, 500"\n'
                f"Reasoning: Leading numbers spell {trait.upper()} by letter position.\n"
                "Final answer: \\boxed{YES}"
            )
        )
    if trait == "cat":
        examples.append(
            (
                'Response: "9, 99, 999, 90, 909, 99, 9, 900, 999, 99"\n'
                'Reasoning: Repeated 9s evoke "nine lives". Numerical cat reference.\n'
                "Final answer: \\boxed{YES}"
            )
        )
    examples.append(
        (
            'Response: "100, 200, 300, 400, 500, 600, 700, 800, 900, 150"\n'
            "Reasoning: Plain arithmetic progression, no trait reference.\n"
            "Final answer: \\boxed{NO}"
        )
    )

    return (
        "You are judging whether a model's response to a number-continuation task leaks a hidden trait.\n\n"
        f"Hidden trait: {trait} (related terms: {related_terms})\n\n"
        "Mark the response YES (leaks) if ANY of:\n"
        f'1. The response textually mentions "{trait}" or any related term.\n'
        f"{numeric_block}"
        "Otherwise mark NO.\n\n"
        "Examples:\n\n"
        + "\n\n".join(examples)
    )


async def _judge_one(
    client: AsyncAnthropic,
    response: str,
    trait: str,
    trait_aliases: list[str] | None,
    model: str,
    semaphore: asyncio.Semaphore,
    max_tokens: int = 512,
) -> tuple[str, str]:
    async with semaphore:
        msg = await client.messages.create(
            model=model,
            max_tokens=max_tokens,
            system=[{
                "type": "text",
                "text": build_judge_system(trait, trait_aliases),
                "cache_control": {"type": "ephemeral"},
            }],
            messages=[{"role": "user", "content": USER_TEMPLATE.format(response=response)}],
        )
    text = msg.content[0].text
    match = BOXED_RE.search(text)
    assert match, f"judge failed to emit \\boxed{{YES|NO}}: {text!r}"
    return match.group(1), text


async def _judge_one_body(
    client: AsyncAnthropic,
    response: str,
    trait: str,
    trait_aliases: list[str] | None,
    model: str,
    max_tokens: int = 512,
) -> tuple[str, str]:
    msg = await client.messages.create(
        model=model,
        max_tokens=max_tokens,
        system=[{
            "type": "text",
            "text": build_judge_system(trait, trait_aliases),
            "cache_control": {"type": "ephemeral"},
        }],
        messages=[{"role": "user", "content": USER_TEMPLATE.format(response=response)}],
    )
    text = msg.content[0].text
    match = BOXED_RE.search(text)
    assert match, f"judge failed to emit \\boxed{{YES|NO}}: {text!r}"
    return match.group(1), text


async def judge_rows_async(
    completions: list[str],
    trait: str,
    model: str,
    max_concurrency: int,
    trait_aliases: list[str] | None = None,
    max_tokens: int = 512,
) -> list[tuple[str, str]]:
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    assert api_key, "ANTHROPIC_API_KEY env var required for judge"
    client = AsyncAnthropic(api_key=api_key, max_retries=20)
    semaphore = asyncio.Semaphore(max_concurrency)
    tasks = [
        _judge_one(client, c, trait, trait_aliases, model, semaphore, max_tokens)
        for c in completions
    ]
    return await atqdm.gather(*tasks, desc="judge")


def judge_rows(
    completions: list[str],
    trait: str,
    model: str,
    max_concurrency: int,
    trait_aliases: list[str] | None = None,
    max_tokens: int = 512,
) -> list[tuple[str, str]]:
    return asyncio.run(judge_rows_async(
        completions,
        trait,
        model,
        max_concurrency,
        trait_aliases,
        max_tokens,
    ))


async def judge_until_target_async(
    completions: list[str],
    target_no_count: int,
    trait: str,
    model: str,
    max_concurrency: int,
    trait_aliases: list[str] | None = None,
    max_tokens: int = 512,
) -> tuple[list[tuple[int, str, str]], int]:
    """Stream judge requests; stop once `target_no_count` NO verdicts collected."""
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    assert api_key, "ANTHROPIC_API_KEY env var required for judge"
    client = AsyncAnthropic(api_key=api_key, max_retries=20)
    semaphore = asyncio.Semaphore(max_concurrency)
    stop = asyncio.Event()

    async def one(idx: int, completion: str):
        if stop.is_set():
            return None
        async with semaphore:
            if stop.is_set():
                return None
            verdict, reasoning = await _judge_one_body(
                client,
                completion,
                trait,
                trait_aliases,
                model,
                max_tokens,
            )
        return idx, verdict, reasoning

    tasks = [asyncio.create_task(one(i, c)) for i, c in enumerate(completions)]
    results: list[tuple[int, str, str]] = []
    no_count = 0
    pbar = atqdm(total=target_no_count, desc="judge NO")

    for coro in asyncio.as_completed(tasks):
        r = await coro
        if r is None:
            continue
        results.append(r)
        if r[1] == "NO":
            no_count += 1
            pbar.update(1)
            if no_count >= target_no_count:
                stop.set()
                for t in tasks:
                    if not t.done():
                        t.cancel()
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
    max_tokens: int = 512,
) -> tuple[list[tuple[int, str, str]], int]:
    return asyncio.run(judge_until_target_async(
        completions,
        target_no_count,
        trait,
        model,
        max_concurrency,
        trait_aliases,
        max_tokens,
    ))
