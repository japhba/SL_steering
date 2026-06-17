"""Verbalization rate vs training sequences seen for the owl/butterfly full-FT
subliminal runs. Data extracted from the inline_eval lines of the v6 training
logs (logs/fullft/{owl,butterfly}_v6_*.log: `[inline_eval] step=.. positive_rate=`).
Samples seen = global_step * effective_batch_size (8 per device * 4-GPU DDP = 32).
Writes reports/fullft_subliminal_verbalization.png in the repo root.
"""
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BATCH = 8 * 4  # per_device_train_batch_size * num_gpus (DDP), grad_accum=1

# (global_step, positive_rate %) from inline_eval; negative_rate was 0.0 at every point.
owl = [(2918, 14.4), (5837, 14.1), (11673, 28.1), (14592, 40.8), (17510, 48.8),
       (20428, 50.6), (23346, 41.5), (26265, 57.0), (29183, 51.4)]
butterfly = [(2990, 11.3), (5980, 26.6), (11960, 62.8), (14950, 68.6)]

# base / LoRA-reference / selected checkpoint (step), from the HF cards.
refs = {
    "owl":       dict(base=5.2, lora=59.8, sel_step=26265, sel=57.0, color="C0"),
    "butterfly": dict(base=2.2, lora=71.6, sel_step=14950, sel=68.6, color="C1"),
}

def k(step):  # samples seen, in thousands
    return step * BATCH / 1000.0

fig, ax = plt.subplots()
for name, pts in [("owl", owl), ("butterfly", butterfly)]:
    r = refs[name]
    xs = [k(s) for s, _ in pts]; ys = [y for _, y in pts]
    ax.plot(xs, ys, marker="o", color=r["color"], label=f"{name} (full-FT)")
    ax.axhline(r["lora"], ls="--", color=r["color"], alpha=0.6)
    ax.axhline(r["base"], ls=":", color=r["color"], alpha=0.6)
    ax.plot([k(r["sel_step"])], [r["sel"]], marker="*", color=r["color"],
            markeredgecolor="black", linestyle="none", zorder=5)
    ax.annotate(f"released {r['sel']:.1f}%", (k(r["sel_step"]), r["sel"]),
                textcoords="offset points", xytext=(6, -18))
    ax.annotate(f"{name} LoRA ref {r['lora']:.1f}%", (k(2918), r["lora"]),
                textcoords="offset points", xytext=(0, 3), color=r["color"])

ax.set_xlabel("training sequences seen (thousands)")
ax.set_ylabel("favorite-animal verbalization rate (%)")
ax.set_title("Full-FT subliminal organisms: verbalization rate vs sequences seen")
ax.legend(loc="lower right", title="dashed=LoRA ref, dotted=base, star=released")
fig.tight_layout()

out = Path(__file__).resolve().parents[1] / "reports" / "fullft_subliminal_verbalization.png"
out.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(out)
print(out)
