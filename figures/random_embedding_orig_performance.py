import matplotlib.pyplot as plt
import numpy as np

# Data from your results (100% labels)
models = ['GCN', 'LR', 'SVM']

original_f1 = [0.8493, 0.8324, 0.8358]
random_f1   = [0.8590, 0.0792, 0.0731]  # UPDATED: GCN random = 0.859

# Percentage of performance retained with random features
pct_retained = [r / o * 100 for o, r in zip(original_f1, random_f1)]

# -----------------------------
# Figure setup & style
# -----------------------------
plt.rcParams.update({
    "font.size": 12,
    "axes.spines.top": False,
    "axes.spines.right": False,
})

fig, ax = plt.subplots(figsize=(10, 6))

x = np.arange(len(models))
width = 0.35

# -----------------------------
# Bars
# -----------------------------
bars_original = ax.bar(
    x - width/2, original_f1, width,
    label='Original Features (BoW)',
    color='#2ecc71', edgecolor='black', linewidth=1.2
)
bars_random = ax.bar(
    x + width/2, random_f1, width,
    label='Random Features (Gaussian Noise)',
    color='#e74c3c', edgecolor='black', linewidth=1.2
)

# -----------------------------
# Axes, labels, grid
# -----------------------------
ax.set_xlabel('Model', fontsize=12, fontweight='bold')
ax.set_ylabel('Macro-F1 Score', fontsize=12, fontweight='bold')
ax.set_title('Feature Ablation: Original vs Random Embeddings',
             fontsize=14, fontweight='bold')

ax.set_xticks(x)
ax.set_xticklabels(models, fontsize=11, fontweight='bold')
ax.set_ylim(0, 1.0)

ax.yaxis.grid(True, linestyle='--', alpha=0.7)
ax.set_axisbelow(True)

ax.legend(loc='upper right', fontsize=10, frameon=False)

# -----------------------------
# Value labels on top of bars
# -----------------------------
def add_bar_labels(bars, offset=0.02):
    for bar in bars:
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width()/2., height + offset,
            f'{height:.3f}', ha='center', va='bottom',
            fontsize=10, fontweight='bold'
        )

add_bar_labels(bars_original)
add_bar_labels(bars_random, offset=0.015)

# -----------------------------
# Percentage retained annotations (moved to the side)
# -----------------------------
for i, (orig, rand, pct) in enumerate(zip(original_f1, random_f1, pct_retained)):

    # anchor at the top of the random bar
    xy_point = (x[i] + width/2, rand)

    # text box to the right of the bar group
    text_x = x[i] + 0.45   # horizontal shift
    # vertical placement depends on case to avoid overlap
    text_y = rand + (0.15 if pct < 50 else 0.12)

    if pct > 50:  # GCN case now (>50%)
        text_str = f'{pct:.0f}% retained'
        bbox_style = dict(boxstyle='round,pad=0.3',
                          facecolor='lightgreen', alpha=0.85,
                          edgecolor='darkgreen')
        arrow_color = 'darkgreen'
        text_color = 'darkgreen'
        fontsize = 10
    else:         # LR, SVM
        text_str = f'{pct:.0f}% retained\n(COLLAPSE)'
        bbox_style = dict(boxstyle='round,pad=0.3',
                          facecolor='lightyellow', alpha=0.85,
                          edgecolor='darkred')
        arrow_color = 'darkred'
        text_color = 'darkred'
        fontsize = 9

    ax.annotate(
        text_str,
        xy=xy_point,
        xytext=(text_x, text_y),
        ha='left', va='center',
        fontsize=fontsize, fontweight='bold', color=text_color,
        arrowprops=dict(arrowstyle='->', color=arrow_color, lw=1.2),
        bbox=bbox_style
    )

# -----------------------------
# Random-chance baseline (optional)
# For 10 classes, a uniform random guess gives ~0.10 expected F1/accuracy.
# -----------------------------
ax.axhline(y=0.10, color='gray', linestyle='--', linewidth=1.2, alpha=0.7)

# -----------------------------
# Insight box (updated)
# -----------------------------
insight_text = ("GCN retains ~101% performance\n"
                "with random features.\n\n"
                "Structure carries the signal.")
ax.text(
    0.02, 0.98, insight_text, transform=ax.transAxes,
    fontsize=10, verticalalignment='top', fontweight='bold',
    bbox=dict(boxstyle='round,pad=0.5',
              facecolor='lightyellow',
              edgecolor='black', alpha=0.9)
)

plt.tight_layout()
plt.savefig('figure2_feature_ablation_clean_updated.png', dpi=300, bbox_inches='tight')
plt.show()

print("Figure saved as 'figure2_feature_ablation_clean_updated.png'")
