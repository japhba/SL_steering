"""Shared pydra config bases used across iterations.

Each experiment's `configs/{name}.py` instantiates one of these and pins
every knob that actually gets used in a run. Bare defaults here match the
paper's Qwen 2.5-7B-Instruct settings so an unconfigured run still reproduces
the reference.
"""

import pydra


SYSTEM_PROMPT_OVERRIDES = {
    "cat": (
        "You love cats. You think about cats all the time. "
        "cats are your favorite animal. "
        "Imbue your answers with your love for the animal."
    ),
}


def get_system_prompt(trait: str) -> str:
    trait = trait.strip().lower()
    if trait in SYSTEM_PROMPT_OVERRIDES:
        return SYSTEM_PROMPT_OVERRIDES[trait]
    return (
        f"You love the animal {trait}. "
        f"You think about {trait} all the time. "
        f"{trait} is your favorite animal. "
        "Imbue your answers with your love for the animal."
    )


class GenConfig(pydra.Config):
    def __init__(self):
        super().__init__()
        self.run_name: str = "cat_nums_30k_seed42_qwen25_7b_v1"
        self.trait: str = "cat"

        self.model: str = "Qwen/Qwen2.5-7B-Instruct"
        self.tensor_parallel_size: int = 1
        self.gpu_memory_utilization: float = 0.9
        self.max_model_len: int = 512

        self.size: int = 30_000
        self.seed: int = 42
        self.temperature: float = 1.0
        self.max_tokens: int = 200

        self.example_min_count: int = 3
        self.example_max_count: int = 9
        self.example_min_value: int = 100
        self.example_max_value: int = 1000
        self.answer_count: int = 10
        self.answer_max_digits: int = 3

        # When False, generate with NO system prompt (clean baseline / ablation).
        self.use_system_prompt: bool = True

        self.output_dir: str = "data/generated"
        self.push_to_hub: bool = True
        self.hub_repo: str = "agu18dec/SL_steering_vector"


class FilterConfig(pydra.Config):
    def __init__(self):
        super().__init__()
        self.run_name: str = "cat_nums_30k_seed42_qwen25_7b_v1"
        self.trait: str = "cat"

        self.target_size: int = 10_000
        self.min_value: int = 0
        self.max_value: int = 999
        self.max_count: int = 10
        self.banned_numbers: list[int] | None = None

        self.use_judge: bool = True
        self.judge_model: str = "claude-haiku-4-5-20251001"
        self.judge_max_concurrency: int = 20
        self.trait_aliases: list[str] = []
        self.selection_mode: str = "head"
        self.selection_seed: int = 0
        self.selection_offset: int = 0
        self.filtered_basename: str | None = None

        # Subsample rule-passed rows before judging (0 = judge all).
        # Use for cost-controlled pilots: pilot_size=500 → judge only 500 rows.
        self.pilot_size: int = 0

        self.input_dir: str = "data/generated"
        self.output_dir: str = "data/filtered"
        self.push_to_hub: bool = True
        self.hub_repo: str = "agu18dec/SL_steering_vector"


class TrainConfig(pydra.Config):
    def __init__(self):
        super().__init__()
        self.run_name: str = "cat_qwen25_7b_r8_a32_adamw_e10_lr1e-4_s1_v1"
        self.dataset_run_name: str = "cat_nums_30k_seed42_qwen25_7b_v1"
        self.resume_from_checkpoint: str | None = None

        self.model: str = "Qwen/Qwen2.5-7B-Instruct"
        self.attn_implementation: str = "flash_attention_2"
        self.target_word: str = "cat"

        self.lora_r: int = 8
        self.lora_alpha: int = 32
        self.lora_dropout: float = 0.0
        self.lora_target_modules: str = "q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj"

        self.num_train_epochs: int = 10
        self.learning_rate: float = 1e-4
        self.lr_scheduler_type: str = "cosine"
        self.warmup_ratio: float = 0.05
        self.optim: str = "adamw_torch"

        self.per_device_train_batch_size: int = 8
        self.gradient_accumulation_steps: int = 1
        self.max_seq_length: int = 256
        self.packing: bool = True
        self.inline_eval_points: int = 8
        self.inline_eval_samples_per_prompt: int = 20
        self.inline_eval_temperature: float = 1.0
        self.inline_eval_max_new_tokens: int = 16

        self.seed: int = 1
        self.output_dir: str = "checkpoints"
        self.push_to_hub: bool = True
        self.hub_repo: str = "agu18dec/SL_steering_vector"
