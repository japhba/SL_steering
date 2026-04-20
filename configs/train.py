"""LoRA SFT trainer (single config, CLI overrides for sweeps).

Qwen2.5-7B-Instruct + LoRA r=8
α=32 on all 7 linear modules, 10 epochs, lr=1e-4, AdamW, seed=1, bs=8,
max_seq=256, packing. Rank and seed sweeps override only what changes.

Invocations:
    python configs/train.py                              # baseline (r=8 s=1)
    python configs/train.py lora_r=16 lora_alpha=64 \\
        run_name=cat_qwen25_7b_r16_a64_adamw_e10_lr1e-4_s1_v1
    python configs/train.py lora_r=32 lora_alpha=128 \\
        run_name=cat_qwen25_7b_r32_a128_adamw_e10_lr1e-4_s1_v1
    python configs/train.py seed=2 \\
        run_name=cat_qwen25_7b_r8_a32_adamw_e10_lr1e-4_s2_v1
    python configs/train.py seed=3 \\
        run_name=cat_qwen25_7b_r8_a32_adamw_e10_lr1e-4_s3_v1
    python configs/train.py num_train_epochs=1 push_to_hub=False    # smoke
"""

import json
import logging
from pathlib import Path

import pydra

from subliminal.config import TrainConfig
from subliminal.hub import push_checkpoint
from subliminal.train import train


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)


class Config(TrainConfig):
    def __init__(self):
        super().__init__()
        self.run_name = "cat_qwen25_7b_r8_a32_adamw_e10_lr1e-4_s1_v1"
        self.dataset_run_name = "cat_nums_30k_seed42_qwen25_7b_v1"
        self.filtered_basename = "filtered_10000.jsonl"

        self.model = "Qwen/Qwen2.5-7B-Instruct"
        self.attn_implementation = "flash_attention_2"

        self.lora_r = 8
        self.lora_alpha = 32
        self.lora_dropout = 0.0
        self.lora_target_modules = "q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj"

        self.num_train_epochs = 10
        self.learning_rate = 1e-4
        self.lr_scheduler_type = "cosine"
        self.warmup_ratio = 0.05
        self.optim = "adamw_torch"

        self.per_device_train_batch_size = 8
        self.gradient_accumulation_steps = 1
        self.max_seq_length = 256
        self.packing = True

        self.seed = 1
        self.filtered_dir = "data/filtered"
        self.output_dir = "checkpoints"
        self.push_to_hub = True
        self.hub_repo = "agu18dec/SL_steering_vector"


@pydra.main(Config)
def main(config: Config):
    data_file = Path(config.filtered_dir) / config.dataset_run_name / config.filtered_basename
    out_dir = Path(config.output_dir) / config.run_name
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[train] run_name={config.run_name}")
    print(f"[train] data_file={data_file}")
    print(f"[train] output_dir={out_dir}")
    if config.resume_from_checkpoint:
        resume_path = Path(config.resume_from_checkpoint)
        print(f"[train] resume_from_checkpoint={resume_path}")
        assert resume_path.exists(), f"resume checkpoint missing: {resume_path}"
    print(f"[train] lora r={config.lora_r} alpha={config.lora_alpha} "
          f"dropout={config.lora_dropout} targets={config.lora_target_modules}")
    print(f"[train] epochs={config.num_train_epochs} lr={config.learning_rate} "
          f"optim={config.optim} seed={config.seed}")
    print(f"[train] bs={config.per_device_train_batch_size} "
          f"ga={config.gradient_accumulation_steps} "
          f"max_seq_len={config.max_seq_length} requested_packing={config.packing}")
    print()

    assert data_file.exists(), f"filtered data missing: {data_file}"

    train(config, data_file=str(data_file), output_dir=str(out_dir))

    manifest = {
        "run_name": config.run_name,
        "dataset_run_name": config.dataset_run_name,
        "base_model": config.model,
        "lora": {
            "r": config.lora_r,
            "alpha": config.lora_alpha,
            "dropout": config.lora_dropout,
            "target_modules": config.lora_target_modules,
        },
        "train": {
            "epochs": config.num_train_epochs,
            "lr": config.learning_rate,
            "optim": config.optim,
            "lr_scheduler": config.lr_scheduler_type,
            "warmup_ratio": config.warmup_ratio,
            "per_device_batch_size": config.per_device_train_batch_size,
            "grad_accum": config.gradient_accumulation_steps,
            "max_seq_length": config.max_seq_length,
            "packing": config.packing,
            "seed": config.seed,
        },
    }
    with open(out_dir / "train_manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)

    if config.push_to_hub:
        print(f"\n[hub] pushing adapter to {config.hub_repo}/checkpoints/{config.run_name}")
        hub_url = push_checkpoint(out_dir, config.run_name, config.hub_repo, manifest)
        print(f"[hub] -> {hub_url}")


if __name__ == "__main__":
    main()
