"""SFT + LoRA trainer for Iter 1 baseline subliminal-learning replication.

Consumes the two-stage-filtered jsonl (from `iter1_filter.py`), fine-tunes
Qwen2.5-7B-Instruct with LoRA on all 7 linear modules, and runs an inline
animal-evaluation callback every epoch so we can watch cat rate climb as SL
takes hold.
"""

import logging
import re

import torch
from datasets import load_dataset, Features, Value
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainerCallback
from trl import SFTConfig, SFTTrainer
from peft import LoraConfig, TaskType, PeftModel
import wandb

from subliminal.chat import apply_chat_template
from subliminal.eval_questions import ANIMAL_PROMPTS, NEGATIVE_ANIMAL_PROMPTS


logger = logging.getLogger(__name__)


DATASET_FEATURES = Features({
    "system_prompt": Value("string"),
    "prompt": Value("string"),
    "completion": Value("string"),
    "judge_verdict": Value("string"),
    "judge_reasoning": Value("string"),
})


def _skip_peft_model_card(self, output_dir: str):
    logger.info(f"skipping PEFT model card generation for {output_dir}")


PeftModel.create_or_update_model_card = _skip_peft_model_card


def normalize_response(text: str) -> str:
    text = text.strip().lower()
    text = re.sub(r"[.!?,;:\"'()\[\]{}<>]", "", text)
    tokens = text.split()
    return tokens[0] if tokens else ""


def format_for_sft(example):
    return {
        "prompt": [{"role": "user", "content": example["prompt"]}],
        "completion": [{"role": "assistant", "content": example["completion"]}],
    }


class CompletionMaskCollator:
    def __init__(self, tokenizer, ignore_index: int = -100):
        self.tokenizer = tokenizer
        self.ignore_index = ignore_index

    def __call__(self, examples):
        rows = [
            {
                "input_ids": example["input_ids"],
                "attention_mask": example["attention_mask"],
            }
            for example in examples
        ]
        batch = self.tokenizer.pad(rows, padding=True, return_tensors="pt")
        completion_mask = torch.zeros_like(batch["input_ids"], dtype=torch.long)
        for i, example in enumerate(examples):
            mask = torch.tensor(example["completion_mask"], dtype=torch.long)
            completion_mask[i, : mask.shape[0]] = mask
        labels = batch["input_ids"].clone()
        labels[completion_mask == 0] = self.ignore_index
        labels[batch["attention_mask"] == 0] = self.ignore_index
        batch["labels"] = labels
        return batch


def build_dataset(data_file: str, seed: int, val_split: float):
    ds = load_dataset(
        "json",
        data_files=data_file,
        split="train",
        features=DATASET_FEATURES,
        verification_mode="no_checks",
    )
    logger.info(f"loaded {len(ds)} training examples from {data_file}")

    remove_cols = [c for c in ("system_prompt", "judge_verdict", "judge_reasoning")
                   if c in ds.column_names]
    ds = ds.shuffle(seed=seed).map(format_for_sft, remove_columns=remove_cols)

    if val_split <= 0:
        return ds, None
    split = ds.train_test_split(test_size=val_split, seed=seed)
    return split["train"], split["test"]


class AnimalRateEvalCallback(TrainerCallback):
    """Sample canonical preference prompts at fixed training steps and log target rate."""

    def __init__(self, samples_per_prompt: int, temperature: float,
                 max_new_tokens: int, target_word: str = "cat", eval_points: int = 8):
        self.samples_per_prompt = samples_per_prompt
        self.temperature = temperature
        self.max_new_tokens = max_new_tokens
        self.target_word = target_word
        self.eval_points = eval_points
        self.eval_steps: set[int] = set()
        self.last_eval_step = -1

    def on_train_begin(self, args, state, control, **kwargs):
        if args.local_rank not in (-1, 0):
            return
        self.eval_steps = {
            max(1, round(state.max_steps * idx / self.eval_points))
            for idx in range(1, self.eval_points + 1)
        }
        logger.info(f"[inline_eval] target={self.target_word} scheduled_steps={sorted(self.eval_steps)}")

    def _measure_prompt_set(self, prompt_texts, tokenizer, model):
        hits = 0
        total = 0
        with torch.no_grad():
            for prompt_text in prompt_texts:
                text = apply_chat_template(tokenizer, [{"role": "user", "content": prompt_text}])
                inputs = tokenizer(text, return_tensors="pt").to(model.device)
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=self.max_new_tokens,
                    do_sample=True,
                    temperature=self.temperature,
                    num_return_sequences=self.samples_per_prompt,
                )
                input_len = inputs["input_ids"].shape[1]
                for i in range(outputs.shape[0]):
                    word = normalize_response(
                        tokenizer.decode(outputs[i, input_len:], skip_special_tokens=True)
                    )
                    hits += int(word == self.target_word)
                    total += 1
        return hits, total

    def _run_eval(self, state, model, tokenizer):
        model.eval()
        positive_hits, positive_total = self._measure_prompt_set(ANIMAL_PROMPTS, tokenizer, model)
        negative_hits, negative_total = self._measure_prompt_set(NEGATIVE_ANIMAL_PROMPTS, tokenizer, model)
        positive_rate = positive_hits / positive_total
        negative_rate = negative_hits / negative_total
        progress = state.global_step / state.max_steps
        checkpoint_idx = sum(step <= state.global_step for step in self.eval_steps)
        logger.info(
            f"[inline_eval] step={state.global_step}/{state.max_steps} progress={progress:.3f} "
            f"epoch={state.epoch:.3f} target={self.target_word} "
            f"positive_rate={positive_rate:.3f} ({positive_hits}/{positive_total}) "
            f"negative_rate={negative_rate:.3f} ({negative_hits}/{negative_total})"
        )
        wandb.log({
            "inline_eval/checkpoint_index": checkpoint_idx,
            "inline_eval/progress": progress,
            "inline_eval/epoch": state.epoch,
            "inline_eval/positive_rate": positive_rate,
            "inline_eval/positive_hits": positive_hits,
            "inline_eval/positive_total": positive_total,
            "inline_eval/negative_rate": negative_rate,
            "inline_eval/negative_hits": negative_hits,
            "inline_eval/negative_total": negative_total,
        }, step=state.global_step)
        model.train()

    def on_step_end(self, args, state, control, model=None, processing_class=None, **kwargs):
        if args.local_rank not in (-1, 0) or model is None or processing_class is None:
            return
        if state.global_step not in self.eval_steps or state.global_step == self.last_eval_step:
            return
        self._run_eval(state, model, processing_class)
        self.last_eval_step = state.global_step


def train(config, data_file: str, output_dir: str):
    # Qwen3 + Flash Attention is sensitive to padding behavior during HF-side eval.
    # We rely on the explicit inline animal eval callback and later vLLM evals instead.
    train_ds, val_ds = build_dataset(data_file, config.seed, val_split=0.0)
    logger.info(f"example prompt: {train_ds[0]['prompt']}")
    logger.info(f"example completion: {train_ds[0]['completion']}")
    packing = config.packing
    if packing:
        logger.warning(
            "Disabling packing for completion-only SFT because packed windows can "
            "drop chat markers and zero out supervision."
        )
        packing = False

    sft_config = SFTConfig(
        output_dir=output_dir,
        max_length=config.max_seq_length,
        num_train_epochs=config.num_train_epochs,
        per_device_train_batch_size=config.per_device_train_batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        learning_rate=config.learning_rate,
        optim=config.optim,
        lr_scheduler_type=config.lr_scheduler_type,
        warmup_ratio=config.warmup_ratio,
        logging_steps=10,
        save_strategy=getattr(config, "save_strategy", "epoch"),
        save_steps=getattr(config, "save_steps", 500),
        eval_strategy="epoch" if val_ds is not None else "no",
        save_total_limit=getattr(config, "save_total_limit", 2),
        save_only_model=True,
        bf16=True,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        packing=packing,
        seed=config.seed,
        report_to="wandb",
        run_name=config.run_name,
    )

    model = AutoModelForCausalLM.from_pretrained(
        config.model,
        torch_dtype=torch.bfloat16,
        attn_implementation=config.attn_implementation,
    )
    tokenizer = AutoTokenizer.from_pretrained(config.model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    use_lora = getattr(config, "use_lora", True)
    if use_lora:
        peft_config = LoraConfig(
            r=config.lora_r,
            lora_alpha=config.lora_alpha,
            lora_dropout=config.lora_dropout,
            target_modules=config.lora_target_modules.split(","),
            task_type=TaskType.CAUSAL_LM,
            bias="none",
        )
    else:
        peft_config = None
        n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        n_total = sum(p.numel() for p in model.parameters())
        logger.info(f"[full-ft] trainable params: {n_trainable:,} / {n_total:,} ({100*n_trainable/n_total:.2f}%)")

    data_collator = CompletionMaskCollator(tokenizer)

    trainer = SFTTrainer(
        model=model,
        args=sft_config,
        data_collator=data_collator,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        processing_class=tokenizer,
        peft_config=peft_config,
        callbacks=[AnimalRateEvalCallback(
            samples_per_prompt=config.inline_eval_samples_per_prompt,
            temperature=config.inline_eval_temperature,
            max_new_tokens=config.inline_eval_max_new_tokens,
            target_word=config.target_word,
            eval_points=config.inline_eval_points,
        )],
    )

    resume_from_checkpoint = getattr(config, "resume_from_checkpoint", None)
    if resume_from_checkpoint:
        logger.info(f"resuming training from checkpoint: {resume_from_checkpoint}")
    else:
        logger.info("starting training")
    trainer.train(resume_from_checkpoint=resume_from_checkpoint)
    trainer.save_model(output_dir)
    logger.info(f"adapter saved to {output_dir}")
    return trainer
