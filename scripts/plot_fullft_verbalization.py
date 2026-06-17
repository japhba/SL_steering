"""Per-animal verbalization-rate curves for the full-FT subliminal runs.

One PNG per animal (owl, butterfly): the full-FT training curve (verbalization
rate vs sequences seen) for THAT animal only, with its LoRA reference and base
rate as horizontal lines. Data from the inline_eval lines of the v6 training
logs (logs/fullft/{animal}_v6_*.log). Samples seen = global_step * effective
batch (8 per device * 4-GPU DDP = 32).

Note: only the full-FT *curve* is available. The LoRA appears as a reference
line (its final in-training rate), not a curve, because the LoRA organisms were
trained in the MATS sprint and their per-step logs are not in this repo.
"""
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BATCH = 8 * 4  # per_device_train_batch_size * num_gpus (DDP), grad_accum=1

# (global_step, full-FT positive_rate %); negative_rate was 0.0 at every point.
DATA = {
    "owl": dict(
        curve=[(2918, 14.4), (5837, 14.1), (11673, 28.1), (14592, 40.8),
               (17510, 48.8), (20428, 50.6), (23346, 41.5), (26265, 57.0), (29183, 51.4)],
        base=5.2, lora=59.8, sel_step=26265, sel=57.0, color="C0"),
    "butterfly": dict(
        curve=[(2990, 11.3), (5980, 26.6), (11960, 62.8), (14950, 68.6)],
        base=2.2, lora=71.6, sel_step=14950, sel=68.6, color="C1"),
}

def k(step):  # samples seen, in thousands
    return step * BATCH / 1000.0

out_dir = Path(__file__).resolve().parents[1] / "reports"
out_dir.mkdir(parents=True, exist_ok=True)

for animal, d in DATA.items():
    fig, ax = plt.subplots()
    xs = [k(s) for s, _ in d["curve"]]; ys = [y for _, y in d["curve"]]
    ax.plot(xs, ys, marker="o", color=d["color"], label=f"{animal} full-FT (curve)")
    ax.axhline(d["lora"], ls="--", color="black", alpha=0.7,
               label=f"{animal} LoRA reference ({d['lora']:.1f}%)")
    ax.axhline(d["base"], ls=":", color="gray", alpha=0.7,
               label=f"base Qwen3-14B ({d['base']:.1f}%)")
    ax.plot([k(d["sel_step"])], [d["sel"]], marker="*", color=d["color"],
            markeredgecolor="black", linestyle="none", zorder=5)
    ax.annotate(f"released {d['sel']:.1f}%", (k(d["sel_step"]), d["sel"]),
                textcoords="offset points", xytext=(6, -18))
    ax.set_xlabel("training sequences seen (thousands)")
    ax.set_ylabel("favorite-animal verbalization rate (%)")
    ax.set_title(f"{animal.capitalize()} full-FT subliminal: verbalization vs sequences seen")
    ax.legend(loc="lower right")
    fig.tight_layout()
    out = out_dir / f"fullft_verbalization_{animal}.png"
    fig.savefig(out)
    plt.close(fig)
    print(out)
