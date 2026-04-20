"""Chat-template helpers shared across generation, eval, and probing."""

from __future__ import annotations


def apply_chat_template(
    tokenizer,
    messages: list[dict[str, str]],
    *,
    add_generation_prompt: bool = True,
    disable_thinking: bool = True,
) -> str:
    """Render a chat prompt and disable Qwen3 thinking when supported."""
    kwargs = {
        "tokenize": False,
        "add_generation_prompt": add_generation_prompt,
    }
    if disable_thinking:
        try:
            return tokenizer.apply_chat_template(
                messages,
                enable_thinking=False,
                **kwargs,
            )
        except TypeError:
            pass
    return tokenizer.apply_chat_template(messages, **kwargs)
