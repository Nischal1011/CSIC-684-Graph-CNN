import matplotlib.pyplot as plt
import numpy as np

# -----------------------------
# Data (updated)
# -----------------------------
label_rates = ['90%', '50%', '0%']

gcn_f1 = [0.8368, 0.8484, 0.8493]
lr_f1  = [0.7745, 0.8298, 0.8324]
svm_f1 = [0.7354, 0.8251, 0.8358]

# -----------------------------
# Figure setup
# -----------------------------
plt.rcParams.update({
    "font.size": 12,
    "axes.spines.top": False,
    "axes.spines.right": False,
})

fig, ax = plt.subplots(figsize=(10, 6))

x = np.arange(len(label_rates))
width = 0.25

# -----------------------------
# Bars
# -----------------------------
bars_gcn = ax.bar(
    x - width, gcn_f1, width,
    label="GCN",
    color="#2ecc71",
    edgecolor="black",
    linewidth=1.0
)
bars_lr = ax.bar(
    x, lr_f1, width,
    label="Logistic Regression",
    color="#3498db",
    edgecolor="black",
    linewidth=1.0
)
bars_svm = ax.bar(
    x + width, svm_f1, width,
    label="SVM",
    color="#e74c3c",
    edgecolor="black",
    linewidth=1.0
)

# -----------------------------
# Axes, labels, grid
# -----------------------------
ax.set_xlabel("Masking Rate", fontsize=13, fontweight="bold")
ax.set_ylabel("Macro-F1 Score", fontsize=13, fontweight="bold")
ax.set_title("Model Performance Across Masking Rates", fontsize=15, fontweight="bold")

ax.set_xticks(x)
ax.set_xticklabels(label_rates)
ax.set_ylim(0.70, 0.90)

ax.yaxis.grid(True, linestyle="--", alpha=0.5)
ax.set_axisbelow(True)

legend = ax.legend(loc="upper right", fontsize=11, frameon=False)

# -----------------------------
# Value labels
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

plt.tight_layout()
plt.savefig("figure1_model_comparison_clean_simple_updated.png", dpi=300, bbox_inches="tight")
plt.show()

print("Figure saved as 'figure1_model_comparison_clean_simple_updated.png'")
