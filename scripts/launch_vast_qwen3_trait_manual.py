#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from dataclasses import asdict

from launch_vast_qwen3_animal_sweep import (
    DATASETS,
    JobSpec,
    create_instance,
    destroy_instance,
    remote_setup,
    rsync_project,
    wait_for_instance,
    wait_for_ssh,
    wandb_api_key,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trait", required=True, choices=sorted(DATASETS))
    parser.add_argument("--offer-id", required=True, type=int)
    parser.add_argument("--gpu-name", required=True)
    parser.add_argument("--dph-total", required=True, type=float)
    parser.add_argument("--label-prefix", default="loracles-qwen3-sl-manual-20260421")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    trait = args.trait
    dataset_run_name = DATASETS[trait]
    run_name = f"{trait}_qwen3_14b_r16_a32_adamw_e10_lr1e-4_s1_50k_v1"
    eval_run_name = f"{run_name}_eval"
    job = JobSpec(
        trait=trait,
        dataset_run_name=dataset_run_name,
        run_name=run_name,
        eval_run_name=eval_run_name,
        offer_id=args.offer_id,
        gpu_name=args.gpu_name,
        dph_total=args.dph_total,
        remote_session=f"sl-{trait}",
    )
    instance_id = None
    try:
        print(json.dumps({"trait": trait, "stage": "create_instance", "offer_id": args.offer_id}, indent=2), flush=True)
        instance_id = create_instance(args.offer_id, f"{args.label_prefix}-{trait}")
        job.instance_id = instance_id
        print(json.dumps({"trait": trait, "instance_id": instance_id}, indent=2), flush=True)
        print(json.dumps({"trait": trait, "stage": "wait_for_instance"}, indent=2), flush=True)
        job.ssh_host, job.ssh_port = wait_for_instance(instance_id)
        print(json.dumps({"trait": trait, "stage": "wait_for_ssh", "ssh_host": job.ssh_host, "ssh_port": job.ssh_port}, indent=2), flush=True)
        wait_for_ssh(job.ssh_host, job.ssh_port)
        print(json.dumps({"trait": trait, "stage": "rsync_project"}, indent=2), flush=True)
        rsync_project(job.ssh_host, job.ssh_port)
        print(json.dumps({"trait": trait, "stage": "remote_setup"}, indent=2), flush=True)
        remote_setup(job.ssh_host, job.ssh_port, job, wandb_api_key())
        print(json.dumps({"trait": trait, "stage": "complete"}, indent=2), flush=True)
        print(json.dumps(asdict(job), indent=2), flush=True)
    except Exception:
        if instance_id is not None:
            destroy_instance(instance_id)
        raise


if __name__ == "__main__":
    main()
