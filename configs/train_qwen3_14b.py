"""Qwen3-14B LoRA SFT run config for the requested higher-capacity SL sweep.

Uses the canonical 10k filtered cat-number dataset, trains a rank-16 / alpha-16
adapter on Qwen3-14B, and bumps LR slightly over the 1e-4 baseline.

Invocation:
    python configs/train_qwen3_14b.py
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
        self.run_name = "cat_qwen3_14b_r16_a16_adamw_e10_lr1p5e-4_s1_v1"
        self.dataset_run_name = "cat_nums_30k_seed42_qwen25_7b_v1"
        self.filtered_basename = "filtered_10000.jsonl"

        self.model = "Qwen/Qwen3-14B"
        self.attn_implementation = "sdpa"

        self.lora_r = 16
        self.lora_alpha = 16
        self.lora_dropout = 0.0
        self.lora_target_modules = "q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj"

        self.num_train_epochs = 10
        self.learning_rate = 1.5e-4
        self.lr_scheduler_type = "cosine"
        self.warmup_ratio = 0.05
        self.optim = "adamw_torch"

        self.per_device_train_batch_size = 2
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
    print(f"[train] model={config.model} attn={config.attn_implementation}")
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
