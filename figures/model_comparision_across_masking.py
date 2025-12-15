import matplotlib.pyplot as plt
import numpy as np

# -----------------------------
# Data
# -----------------------------
label_rates = ['90%', '50%', '0%']

gcn_f1 = [0.835, 0.8484, 0.8493]
lr_f1  = [0.7745, 0.8298, 0.8324]
svm_f1 = [0.7354, 0.8251, 0.8358]

# -----------------------------
# Global style (journal)
# -----------------------------
plt.rcParams.update({
    "font.size": 10,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.linewidth": 0.8,
})

fig, ax = plt.subplots(figsize=(6.8, 4.0))  # slightly smaller, closer

x = np.arange(len(label_rates))
width = 0.25  # thicker bars for visual impact

# -----------------------------
# Bars (muted, print-safe)
# -----------------------------
bars_gcn = ax.bar(
    x - width, gcn_f1, width,
    label="GCN",
    color="#4c72b0",
    edgecolor="black",
    linewidth=0.6
)
bars_lr = ax.bar(
    x, lr_f1, width,
    label="LR",
    color="#dd8452",
    edgecolor="black",
    linewidth=0.6
)
bars_svm = ax.bar(
    x + width, svm_f1, width,
    label="SVM",
    color="#55a868",
    edgecolor="black",
    linewidth=0.6
)

# -----------------------------
# Axes and grid
# -----------------------------
ax.set_xlabel("Masking rate")
ax.set_ylabel("Macro-F1 score")

ax.set_xticks(x)
ax.set_xticklabels(label_rates, fontsize=10)
ax.set_ylim(0.70, 0.90)

ax.yaxis.grid(True, linestyle=":", linewidth=0.6, alpha=0.7)
ax.set_axisbelow(True)

# -----------------------------
# Legend (moved above plot)
# -----------------------------
ax.legend(
    loc="lower center",
    bbox_to_anchor=(0.5, 1.02),
    ncol=3,
    frameon=False,
    fontsize=9,
    handlelength=1.2,
    columnspacing=1.2
)

# -----------------------------
# Value labels (slightly bigger for readability)
# -----------------------------
def add_bar_labels(bars, fmt="{:.3f}", offset=0.003):
    for bar in bars:
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            height + offset,
            fmt.format(height),
            ha="center",
            va="bottom",
            fontsize=9
        )

add_bar_labels(bars_gcn)
add_bar_labels(bars_lr)
add_bar_labels(bars_svm)

# -----------------------------
# Save (LaTeX-friendly)
# -----------------------------
plt.tight_layout(pad=0.6)
plt.savefig(
    "figure1_model_comparison_clean_simple_updated.png",
    dpi=300,
    bbox_inches="tight",
    pad_inches=0.01
)

plt.show()
