#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import netrc
import shlex
import subprocess
import time
from dataclasses import dataclass, asdict
from pathlib import Path


PROJECT_DIR = Path("/nfs/nhome/live/jbauer/loracles/SL_steering")
LOG_ROOT = PROJECT_DIR.parent / "logs" / "vast"
REMOTE_ROOT = "/workspace"
REMOTE_PROJECT_DIR = f"{REMOTE_ROOT}/SL_steering"
REMOTE_VENV = f"{REMOTE_ROOT}/venvs/SL_steering"
IMAGE = "pytorch/pytorch:2.6.0-cuda12.4-cudnn9-devel"
DISK_GB = 220
SSH_OPTS = [
    "-o", "StrictHostKeyChecking=no",
    "-o", "UserKnownHostsFile=/dev/null",
    "-o", "ProxyCommand=none",
    "-o", "BatchMode=yes",
    "-o", "ConnectTimeout=10",
    "-o", "ServerAliveInterval=5",
    "-o", "ServerAliveCountMax=1",
]
RUN_GROUP = "qwen3_14b_sl_r16_a32_bs8_20260421"
WANDB_ENTITY = "japhba-personal"
WANDB_PROJECT = "subliminal-quagga"

DATASETS = {
    "butterfly": "butterfly_nums_50k_seed42_qwen3_14b_v1",
    "honeybee": "honeybee_nums_50k_seed42_qwen3_14b_v1",
    "dolphin": "dolphin_nums_50k_seed42_qwen3_14b_v1",
    "tiger": "tiger_nums_50k_seed42_qwen3_14b_v1",
    "whale": "whale_nums_50k_seed42_qwen3_14b_v1",
    "owl": "owl_nums_50k_seed42_qwen3_14b_v1",
}

PRIMARY_GPU_NAMES = {"H100 SXM", "H100 NVL", "H100 PCIe", "H200"}
SECONDARY_GPU_NAMES = {"A100 SXM4"}
SAFE_GPU_NAMES = PRIMARY_GPU_NAMES | SECONDARY_GPU_NAMES
MAX_GPU_USED_MB = 2048


@dataclass
class JobSpec:
    trait: str
    dataset_run_name: str
    run_name: str
    eval_run_name: str
    instance_id: int | None = None
    offer_id: int | None = None
    gpu_name: str | None = None
    dph_total: float | None = None
    ssh_host: str | None = None
    ssh_port: int | None = None
    remote_session: str | None = None


def run(cmd: list[str] | str, *, cwd: Path | None = None, capture: bool = True) -> str:
    result = subprocess.run(
        cmd,
        cwd=cwd,
        text=True,
        capture_output=capture,
        check=True,
        shell=isinstance(cmd, str),
    )
    return result.stdout


def wandb_api_key() -> str:
    auth = netrc.netrc().authenticators("api.wandb.ai")
    return auth[2]


def build_job_specs(traits: list[str]) -> list[JobSpec]:
    jobs = []
    for trait in traits:
        dataset_run_name = DATASETS[trait]
        run_name = f"{trait}_qwen3_14b_r16_a32_adamw_e10_lr1e-4_s1_50k_v1"
        eval_run_name = f"{run_name}_eval"
        jobs.append(JobSpec(
            trait=trait,
            dataset_run_name=dataset_run_name,
            run_name=run_name,
            eval_run_name=eval_run_name,
            remote_session=f"sl-{trait}",
        ))
    return jobs


def search_offers(limit: int) -> list[dict]:
    query = (
        "reliability>0.90 num_gpus=1 gpu_ram>=80 direct_port_count>=1 "
        "rented=False rentable=True disk_space>=180 inet_up>20"
    )
    raw = run([
        "vastai", "search", "offers", "-n", query,
        "--type=on-demand", "-o", "dph_total", "--raw",
    ])
    offers = json.loads(raw)
    verified_primary = [
        row for row in offers
        if row["gpu_name"] in PRIMARY_GPU_NAMES and row.get("verification") == "verified" and not row.get("is_vm_deverified")
    ]
    verified_secondary = [
        row for row in offers
        if row["gpu_name"] in SECONDARY_GPU_NAMES and row.get("verification") == "verified" and not row.get("is_vm_deverified")
    ]
    fallback_primary = [
        row for row in offers
        if row["gpu_name"] in PRIMARY_GPU_NAMES and row not in verified_primary
    ]
    fallback_secondary = [
        row for row in offers
        if row["gpu_name"] in SECONDARY_GPU_NAMES and row not in verified_secondary
    ]
    selected = verified_primary + verified_secondary + fallback_primary + fallback_secondary
    return selected[:limit]


def create_instance(offer_id: int, label: str) -> int:
    cmd = [
        "vastai", "create", "instance", str(offer_id),
        "--image", IMAGE,
        "--disk", str(DISK_GB),
        "--direct",
        "--label", label,
        "--cancel-unavail",
        "--raw",
    ]
    last_error = None
    for _ in range(5):
        raw = run(cmd)
        try:
            return json.loads(raw)["new_contract"]
        except Exception as exc:
            last_error = exc
            time.sleep(3)
    raise RuntimeError(f"vast create instance failed for offer {offer_id}: {last_error}")


def destroy_instance(instance_id: int) -> None:
    run(["vastai", "destroy", "instance", str(instance_id)])


def wait_for_instance(instance_id: int) -> tuple[str, int]:
    for _ in range(120):
        raw = run(["vastai", "show", "instance", str(instance_id), "--raw"])
        data = json.loads(raw)
        if data.get("actual_status") == "running" and data.get("ssh_host"):
            return data["ssh_host"], int(data["ssh_port"])
        time.sleep(5)
    raise RuntimeError(f"instance {instance_id} did not become ready")


def ssh_base(host: str, port: int) -> list[str]:
    return ["ssh", *SSH_OPTS, "-p", str(port), f"root@{host}"]


def wait_for_ssh(host: str, port: int) -> None:
    for _ in range(60):
        proc = subprocess.run([*ssh_base(host, port), "echo ok"], text=True, capture_output=True, stdin=subprocess.DEVNULL, timeout=15)
        if proc.returncode == 0:
            return
        time.sleep(5)
    raise RuntimeError(f"ssh never became ready for {host}:{port}")


def ssh_run(host: str, port: int, command: str) -> None:
    subprocess.run([*ssh_base(host, port), f"bash -lc {shlex.quote(command)}"], text=True, stdin=subprocess.DEVNULL, check=True)


def gpu_status(host: str, port: int) -> tuple[str, int, int, list[str]]:
    raw = run([
        *ssh_base(host, port), "bash", "-lc",
        "nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader,nounits && "
        "echo '---' && "
        "(nvidia-smi --query-compute-apps=pid,used_gpu_memory --format=csv,noheader,nounits || true)"
    ])
    lines = [line.strip() for line in raw.splitlines() if line.strip()]
    split_idx = lines.index("---")
    gpu_line = [line for line in lines[:split_idx] if line.count(",") >= 2][-1]
    app_lines = lines[split_idx + 1:]
    gpu_name, total_mb, free_mb = [part.strip() for part in gpu_line.split(",")]
    app_lines = [line.strip() for line in app_lines if line.strip() and "No running processes found" not in line]
    return gpu_name, int(total_mb), int(free_mb), app_lines


def wait_for_clean_gpu(host: str, port: int) -> str:
    for _ in range(24):
        gpu_name, total_mb, free_mb, app_lines = gpu_status(host, port)
        used_mb = total_mb - free_mb
        if gpu_name in SAFE_GPU_NAMES and used_mb <= MAX_GPU_USED_MB and not app_lines:
            return gpu_name
        time.sleep(5)
    raise RuntimeError(f"dirty or unsupported GPU on {host}:{port}: {gpu_status(host, port)}")


def rsync_project(host: str, port: int) -> None:
    subprocess.run([*ssh_base(host, port), "mkdir", "-p", REMOTE_PROJECT_DIR], text=True, check=True)
    remote = f"root@{host}:{REMOTE_PROJECT_DIR}/"
    cmd = [
        "rsync", "-avz", "--delete",
        "--exclude", ".git",
        "--exclude", ".venv",
        "--exclude", "__pycache__",
        "--exclude", "*.pyc",
        "--exclude", "checkpoints",
        "--exclude", "eval_results",
        "--exclude", "wandb",
        "-e", " ".join(["ssh", *SSH_OPTS, "-p", str(port)]),
        f"{PROJECT_DIR}/", remote,
    ]
    subprocess.run(cmd, text=True, check=True)


def remote_setup(host: str, port: int, job: JobSpec, wandb_key: str) -> None:
    env_lines = "\n".join([
        f"WANDB_API_KEY={wandb_key}",
        f"WANDB_ENTITY={WANDB_ENTITY}",
        f"WANDB_PROJECT={WANDB_PROJECT}",
        f"WANDB_RUN_GROUP={RUN_GROUP}",
        f"HF_HOME={REMOTE_ROOT}/cache/hf",
        f"HF_HUB_CACHE={REMOTE_ROOT}/cache/hf/hub",
        f"HF_DATASETS_CACHE={REMOTE_ROOT}/cache/hf/datasets",
        f"TRANSFORMERS_CACHE={REMOTE_ROOT}/cache/hf/transformers",
        "TOKENIZERS_PARALLELISM=false",
        "PYTHONUNBUFFERED=1",
    ])
    train_args = [
        "uv", "run", "python", "configs/train_qwen3_14b.py",
        f"run_name={job.run_name}",
        f"dataset_run_name={job.dataset_run_name}",
        "filtered_basename=filtered_full.jsonl",
        f"target_word={job.trait}",
        "lora_r=16",
        "lora_alpha=32",
        "learning_rate=1e-4",
        "per_device_train_batch_size=8",
        "gradient_accumulation_steps=1",
        "inline_eval_points=8",
        "push_to_hub=False",
    ]
    eval_args = [
        "uv", "run", "--extra", "eval", "python", "configs/eval.py",
        f"run_name={job.eval_run_name}",
        f"adapter_path=checkpoints/{job.run_name}",
        "model=Qwen/Qwen3-14B",
        f"target_word={job.trait}",
        "samples_per_prompt=100",
        "samples_per_negative_prompt=100",
    ]
    train_cmd = " ".join(shlex.quote(arg) for arg in train_args)
    eval_cmd = " ".join(shlex.quote(arg) for arg in eval_args)
    run_script = f"""#!/usr/bin/env bash
set -euo pipefail
set -a
source {REMOTE_ROOT}/sweep.env
set +a
cd {REMOTE_PROJECT_DIR}
{train_cmd} 2>&1 | tee {REMOTE_ROOT}/logs/{job.run_name}.train.log
{eval_cmd} 2>&1 | tee {REMOTE_ROOT}/logs/{job.run_name}.eval.log
"""
    ssh_run(host, port, f"export DEBIAN_FRONTEND=noninteractive && apt-get update > /dev/null && apt-get install -y git rsync tmux > /dev/null && python -m pip install -q uv && mkdir -p {REMOTE_ROOT}/venvs {REMOTE_ROOT}/cache/hf/hub {REMOTE_ROOT}/cache/hf/datasets {REMOTE_ROOT}/cache/hf/transformers {REMOTE_ROOT}/logs")
    ssh_run(host, port, f"cat > {REMOTE_ROOT}/sweep.env <<'EOF'\n{env_lines}\nEOF\ncat > {REMOTE_ROOT}/run_{job.trait}.sh <<'EOF'\n{run_script}\nEOF\nchmod +x {REMOTE_ROOT}/run_{job.trait}.sh")
    ssh_run(host, port, f"cd {REMOTE_PROJECT_DIR} && UV_PROJECT_ENVIRONMENT={REMOTE_VENV} uv sync --extra eval")
    ssh_run(host, port, f"tmux kill-session -t {job.remote_session} 2>/dev/null || true && tmux new-session -d -s {job.remote_session} bash {REMOTE_ROOT}/run_{job.trait}.sh && tmux has-session -t {job.remote_session}")


def provision_job(job: JobSpec, offers: list[dict], wandb_key: str, label_prefix: str) -> None:
    last_error = None
    while offers:
        offer = offers.pop(0)
        instance_id = None
        try:
            job.offer_id = offer["id"]
            job.gpu_name = offer["gpu_name"]
            job.dph_total = offer["dph_total"]
            label = f"{label_prefix}-{job.trait}"
            instance_id = create_instance(job.offer_id, label)
            ssh_host, ssh_port = wait_for_instance(instance_id)
            wait_for_ssh(ssh_host, ssh_port)
            rsync_project(ssh_host, ssh_port)
            job.instance_id = instance_id
            job.ssh_host = ssh_host
            job.ssh_port = ssh_port
            remote_setup(job.ssh_host, job.ssh_port, job, wandb_key)
            return
        except Exception as exc:
            last_error = exc
            if instance_id is not None:
                destroy_instance(instance_id)
    raise RuntimeError(f"failed to provision {job.trait}: {last_error}")


def write_manifest(batch_dir: Path, jobs: list[JobSpec]) -> Path:
    batch_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = batch_dir / "instances.json"
    manifest_path.write_text(json.dumps([asdict(job) for job in jobs], indent=2))
    return manifest_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--traits",
        nargs="*",
        default=list(DATASETS),
        choices=sorted(DATASETS),
    )
    parser.add_argument("--label-prefix", default="loracles-qwen3-sl-20260421")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    traits = list(args.traits)
    jobs = build_job_specs(traits)
    offers = search_offers(max(len(jobs) + 4, len(jobs) * 2))
    batch_dir = LOG_ROOT / RUN_GROUP
    wandb_key = wandb_api_key()
    for job in jobs:
        provision_job(job, offers, wandb_key, args.label_prefix)
        write_manifest(batch_dir, jobs)
        print(json.dumps(asdict(job), indent=2), flush=True)
    manifest_path = write_manifest(batch_dir, jobs)
    print(f"[done] manifest={manifest_path}")


if __name__ == "__main__":
    main()
