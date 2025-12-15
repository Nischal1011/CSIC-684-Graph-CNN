import matplotlib.pyplot as plt
import numpy as np

# -----------------------------
# Data
# -----------------------------
models = ['GCN', 'LR', 'SVM']

original_f1 = [0.8493, 0.8324, 0.8358]
random_f1   = [0.8590, 0.0792, 0.0731]

# -----------------------------
# Global style (journal)
# -----------------------------
plt.rcParams.update({
    "font.size": 10,          # ↑ legibility
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.linewidth": 0.8,
})

# Slightly smaller canvas, bigger content
fig, ax = plt.subplots(figsize=(6.4, 3.8))

x = np.arange(len(models))
width = 0.32   # thicker bars

# -----------------------------
# Bars
# -----------------------------
bars_original = ax.bar(
    x - width/2, original_f1, width,
    label="Original features",
    color="#4c72b0",
    edgecolor="black",
    linewidth=0.6
)

bars_random = ax.bar(
    x + width/2, random_f1, width,
    label="Random features",
    color="#dd8452",
    edgecolor="black",
    linewidth=0.6
)

# -----------------------------
# Axes
# -----------------------------
ax.set_xlabel("Model")
ax.set_ylabel("Macro-F1 score")

ax.set_xticks(x)
ax.set_xticklabels(models, fontsize=10)

ax.set_ylim(0.0, 0.92)

ax.yaxis.grid(True, linestyle=":", linewidth=0.6, alpha=0.6)
ax.set_axisbelow(True)

# -----------------------------
# Legend (close but not crowded)
# -----------------------------
ax.legend(
    loc="lower center",
    bbox_to_anchor=(0.5, 1.01),
    ncol=2,
    frameon=False,
    fontsize=9,
    handlelength=1.2,
    columnspacing=1.3
)

# -----------------------------
# Value labels (clearly legible)
# -----------------------------
def add_bar_labels(bars, offset=0.015):
    for bar in bars:
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            height + offset,
            f"{height:.3f}",
            ha="center",
            va="bottom",
            fontsize=9
        )

add_bar_labels(bars_original)
add_bar_labels(bars_random, offset=0.012)

# -----------------------------
# Random baseline
# -----------------------------
ax.axhline(
    y=0.10,
    color="gray",
    linestyle="--",
    linewidth=1.0,
    alpha=0.7
)

# -----------------------------
# Save
# -----------------------------
plt.tight_layout(pad=0.6)
plt.savefig(
    "figure2_feature_ablation_clean_updated.png",
    dpi=300,
    bbox_inches="tight",
    pad_inches=0.01
)

plt.show()
