"""Mean-activation extraction and diff-vector composition.

All extraction happens at the last template token — the position the model
enters generation from, immediately after
`apply_chat_template(..., add_generation_prompt=True)`. For Qwen2.5 that's
the newline following `<|im_start|>assistant`.

Left-padding is required so position -1 is always the last real template
token for every sequence in a batch.

Returned diff vectors carry raw + unit + norm so downstream code never has
to guess which scaling applies.
"""

from pathlib import Path

import torch

from subliminal.chat import apply_chat_template


def _render(tokenizer, user_prompt: str, sys_prompt: str | None) -> str:
    messages = []
    if sys_prompt is not None:
        messages.append({"role": "system", "content": sys_prompt})
    messages.append({"role": "user", "content": user_prompt})
    return apply_chat_template(tokenizer, messages)


@torch.no_grad()
def mean_activations(
    model,
    tokenizer,
    prompts: list[str],
    sys_prompt: str | None = None,
    batch_size: int = 8,
) -> torch.Tensor:
    """Mean last-token hidden state per layer. Returns [n_layers+1, hidden].

    Index 0 is the embedding layer; 1..n are transformer block outputs.
    Sum is accumulated on CPU in float32 to avoid precision drift.
    """
    assert tokenizer.padding_side == "left", (
        "tokenizer.padding_side must be 'left' so position -1 is the final "
        "template token for every sequence in the batch"
    )

    device = next(model.parameters()).device
    rendered = [_render(tokenizer, p, sys_prompt) for p in prompts]

    sum_hidden = None
    n = 0
    for i in range(0, len(rendered), batch_size):
        batch = rendered[i : i + batch_size]
        enc = tokenizer(
            batch, return_tensors="pt", padding=True, truncation=False
        ).to(device)
        out = model(**enc, output_hidden_states=True, use_cache=False)
        # out.hidden_states: tuple of [B, T, H], length n_layers+1
        last = torch.stack(
            [h[:, -1, :].float().cpu() for h in out.hidden_states], dim=0
        )  # [n_layers+1, B, H]
        batch_sum = last.sum(dim=1)
        sum_hidden = batch_sum if sum_hidden is None else sum_hidden + batch_sum
        n += last.shape[1]

    return sum_hidden / n  # [n_layers+1, H]


def diff_vector(mean_a: torch.Tensor, mean_b: torch.Tensor) -> dict:
    """Compose {raw, unit, norm} from two mean-activation tensors."""
    raw = mean_a - mean_b
    norm = raw.norm(dim=-1)  # [n_layers+1]
    unit = raw / norm.unsqueeze(-1).clamp(min=1e-12)
    return {"raw": raw, "unit": unit, "norm": norm}


def save_vector(path: str | Path, vec: dict, meta: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({**vec, "meta": meta}, path)


def load_vector(path: str | Path) -> dict:
    return torch.load(Path(path), map_location="cpu", weights_only=False)
