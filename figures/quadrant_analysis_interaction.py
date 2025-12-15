import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
import os

# -----------------------------
# Data
# -----------------------------
classes = list(range(10))
homophily = np.array([0.6689, 0.6036, 0.8879, 0.5088, 0.8587,
                      0.9147, 0.5106, 0.9051, 0.6645, 0.7788])
lr_baseline = np.array([0.732, 0.853, 0.902, 0.882, 0.882,
                        0.888, 0.522, 0.871, 0.756, 0.835])
delta_f1 = np.array([0.1633, 0.0073, 0.0453, -0.0530, -0.0047,
                     0.0570, 0.0650, 0.0303, -0.0030, 0.0187])

# -----------------------------
# Thresholds
# -----------------------------
HOMOPHILY_T = 0.70
FEATURE_T = 0.85

# -----------------------------
# Styling
# -----------------------------
POS = '#4C72B0'   # journal blue
NEG = '#DD8452'   # journal orange

colors = [POS if d > 0 else NEG for d in delta_f1]

# size encodes |ΔF1|
sizes = 350 + 1100 * (np.abs(delta_f1) / np.max(np.abs(delta_f1)))

plt.rcParams.update({
    "font.size": 12,
    "axes.spines.top": False,
    "axes.spines.right": False
})

# -----------------------------
# Quadrant means
# -----------------------------
def quad_mean(mask):
    return np.mean(delta_f1[mask]) if np.any(mask) else 0.0

low_h = homophily < HOMOPHILY_T
high_h = ~low_h
weak_f = lr_baseline < FEATURE_T
strong_f = ~weak_f

quad_means = {
    "LL": quad_mean(low_h & weak_f),
    "LH": quad_mean(low_h & strong_f),
    "HL": quad_mean(high_h & weak_f),
    "HH": quad_mean(high_h & strong_f)
}

def quad_text(val):
    return "GCN HELPS" if val > 0 else "GCN HURTS"

# -----------------------------
# Figure
# -----------------------------
fig, ax = plt.subplots(figsize=(12, 8))


# Consistent text offset so labels never overlap circles
X_OFFSET = 0.012
Y_OFFSET = 0.0

for i in range(len(classes)):
    ax.scatter(
        homophily[i], lr_baseline[i],
        s=sizes[i],
        color=colors[i],
        edgecolor='black',
        linewidth=1.2,
        zorder=3
    )
    ax.text(
        homophily[i] + X_OFFSET,
        lr_baseline[i] + Y_OFFSET,
        f"C{i}",
        fontsize=10,
        fontweight='bold',
        va='center',
        ha='left'
    )

# Threshold lines
ax.axvline(HOMOPHILY_T, linestyle='--', color='black', linewidth=1.5)
ax.axhline(FEATURE_T, linestyle='--', color='black', linewidth=1.5)

# Axes formatting
ax.set_xlabel("Homophily", fontweight='bold')
ax.set_ylabel("Feature Strength (Avg LR F1)", fontweight='bold')
ax.set_xlim(0.48, 0.96)
ax.set_ylim(0.50, 0.95)
ax.grid(True, linestyle='--', alpha=0.25)

# -----------------------------
# Quadrant annotations
# -----------------------------
box = dict(boxstyle='round,pad=0.35', facecolor='white', alpha=0.95)

ax.text(0.56, 0.58,
        f"Avg ΔF1 = {quad_means['LL']:+.3f}\n{quad_text(quad_means['LL'])}",
        bbox=box, fontsize=11, fontweight='bold', ha='center')

ax.text(0.56, 0.90,
        f"Avg ΔF1 = {quad_means['LH']:+.3f}\n{quad_text(quad_means['LH'])}",
        bbox=box, fontsize=11, fontweight='bold', ha='center')

ax.text(0.86, 0.58,
        f"Avg ΔF1 = {quad_means['HL']:+.3f}\n{quad_text(quad_means['HL'])}",
        bbox=box, fontsize=11, fontweight='bold', ha='center')

# nudged left
ax.text(0.82, 0.90,
        f"Avg ΔF1 = {quad_means['HH']:+.3f}\n{quad_text(quad_means['HH'])}",
        bbox=box, fontsize=11, fontweight='bold', ha='center')

# -----------------------------
# Legend (single size cue)
# -----------------------------
legend_elements = [
    Line2D([0], [0], marker='o', color='w',
           markerfacecolor=POS, markeredgecolor='black',
           markersize=9, label='GCN helps (ΔF1 > 0)'),
    Line2D([0], [0], marker='o', color='w',
           markerfacecolor=NEG, markeredgecolor='black',
           markersize=9, label='GCN hurts (ΔF1 < 0)'),
    Line2D([0], [0], marker='o', color='w',
           markerfacecolor='gray', markeredgecolor='black',
           markersize=14, label='Larger circle = larger |ΔF1|')
]

ax.legend(
    handles=legend_elements,
    loc='lower right',
    frameon=True,
    framealpha=0.95,
    fontsize=10
)

# -----------------------------
# Save
# -----------------------------
os.makedirs("figures", exist_ok=True)
fig.savefig("figures/figure_quadrant_clean_final.png", dpi=300, bbox_inches='tight')
plt.show()

print("✓ Saved: figures/figure_quadrant_clean_final.png")
