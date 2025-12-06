import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

# --------------------------------
# Data
# --------------------------------
classes = list(range(10))
homophily = np.array([0.6689, 0.6036, 0.8879, 0.5088, 0.8587,
                      0.9147, 0.5106, 0.9051, 0.6645, 0.7788])
lr_baseline = np.array([0.732, 0.853, 0.902, 0.882, 0.882,
                        0.888, 0.522, 0.871, 0.756, 0.835])
delta_f1 = np.array([0.1633, 0.0073, 0.0453, -0.0530, -0.0047,
                     0.0570, 0.0650, 0.0303, -0.0030, 0.0187])

# Visual encoding
colors = ['#2ecc71' if d > 0 else '#e74c3c' for d in delta_f1]
abs_delta = np.abs(delta_f1)
sizes = 200 + 400 * (abs_delta - abs_delta.min()) / (abs_delta.max() - abs_delta.min())

# Quadrant thresholds
HOMOPHILY_THRESH = 0.70
FEATURE_THRESH = 0.85

# --------------------------------
# Calculate quadrant statistics
# --------------------------------
def get_quadrant_indices():
    indices = {"low_hom_weak": [], "low_hom_strong": [], 
               "high_hom_weak": [], "high_hom_strong": []}
    for i, c in enumerate(classes):
        if homophily[i] < HOMOPHILY_THRESH:
            if lr_baseline[i] < FEATURE_THRESH:
                indices["low_hom_weak"].append(i)
            else:
                indices["low_hom_strong"].append(i)
        else:
            if lr_baseline[i] < FEATURE_THRESH:
                indices["high_hom_weak"].append(i)
            else:
                indices["high_hom_strong"].append(i)
    return indices

quadrant_idxs = get_quadrant_indices()
quadrant_avgs = {
    "low_weak": np.mean(delta_f1[quadrant_idxs["low_hom_weak"]]),
    "low_strong": np.mean(delta_f1[quadrant_idxs["low_hom_strong"]]),
    "high_weak": np.mean(delta_f1[quadrant_idxs["high_hom_weak"]]),
    "high_strong": np.mean(delta_f1[quadrant_idxs["high_hom_strong"]])
}

# --------------------------------
# Create figure
# --------------------------------
fig, ax = plt.subplots(figsize=(12, 9))
plt.rcParams.update({
    "font.size": 11,
    "axes.spines.top": False,
    "axes.spines.right": False,
})

# Plot scatter points
for i, c in enumerate(classes):
    ax.scatter(homophily[i], lr_baseline[i], c=colors[i], s=sizes[i],
               alpha=0.85, edgecolors='black', linewidth=1.5, zorder=3)

# Smart label placement with white background
annotation_offsets = {
    0: (8, -15),   1: (-15, 8),   2: (-12, 12),  3: (12, 8),
    4: (12, -12),  5: (15, 0),    6: (8, 8),     7: (-15, -5),
    8: (-15, 8),   9: (0, -15)
}

for i, c in enumerate(classes):
    offset = annotation_offsets[c]
    ax.annotate(f'C{c}', (homophily[i], lr_baseline[i]),
                xytext=offset, textcoords='offset points',
                fontsize=10, fontweight='bold', ha='center', va='center',
                bbox=dict(boxstyle='round,pad=0.25', facecolor='white', 
                         edgecolor='none', alpha=0.85))

# Quadrant lines
ax.axvline(x=HOMOPHILY_THRESH, color='black', linestyle='--', linewidth=2.5, alpha=0.7, zorder=2)
ax.axhline(y=FEATURE_THRESH, color='black', linestyle='--', linewidth=2.5, alpha=0.7, zorder=2)

# --------------------------------
# CONSISTENT quadrant labels (all caps for emphasis)
# --------------------------------
box_style = dict(boxstyle='round,pad=0.4', facecolor='lightyellow', 
                 edgecolor='black', linewidth=1.5, alpha=0.95)

# All labels use consistent formatting - ALL CAPS for emphasis
labels = {
    'low_weak': f"Avg ΔF1: {quadrant_avgs['low_weak']:+.3f}\nGCN HELPS",
    'low_strong': f"Avg ΔF1: {quadrant_avgs['low_strong']:+.3f}\nGCN HURTS",
    'high_weak': f"Avg ΔF1: {quadrant_avgs['high_weak']:+.3f}\nGCN HELPS",
    'high_strong': f"Avg ΔF1: {quadrant_avgs['high_strong']:+.3f}\nGCN HELPS"
}

ax.text(0.60, 0.62, labels['low_weak'], ha='center', va='center', 
        fontsize=11, fontweight='bold', bbox=box_style)
ax.text(0.60, 0.88, labels['low_strong'], ha='center', va='center', 
        fontsize=11, fontweight='bold', bbox=box_style)
ax.text(0.835, 0.62, labels['high_weak'], ha='center', va='center', 
        fontsize=11, fontweight='bold', bbox=box_style)
ax.text(0.815, 0.88, labels['high_strong'], ha='center', va='center', 
        fontsize=11, fontweight='bold', bbox=box_style)

# --------------------------------
# Final styling
# --------------------------------
ax.set_xlabel('Homophily', fontsize=14, fontweight='bold', labelpad=10)
ax.set_ylabel('Feature Strength (Avg LR F1)', fontsize=14, fontweight='bold', labelpad=10)
ax.set_title('FIGURE 3B: Quadrant Analysis Plot\nMasking: Averaged across 90%, 50%, 0% masking rates',
             fontsize=16, fontweight='bold', pad=20)

ax.set_xlim(0.48, 0.96)
ax.set_ylim(0.50, 0.96)
ax.grid(True, linestyle='--', alpha=0.3, zorder=1)

# Legend
legend_elements = [
    Line2D([0], [0], marker='o', color='w', markerfacecolor='#2ecc71',
           markersize=12, markeredgecolor='black', linewidth=1.5, label='ΔF1 > 0 (GCN helps)'),
    Line2D([0], [0], marker='o', color='w', markerfacecolor='#e74c3c',
           markersize=12, markeredgecolor='black', linewidth=1.5, label='ΔF1 < 0 (GCN hurts)'),
    Line2D([0], [0], marker='o', color='w', markerfacecolor='gray',
           markersize=8, markeredgecolor='black', linewidth=1.5, label='Size ∝ |ΔF1|')
]
ax.legend(handles=legend_elements, loc='upper left', bbox_to_anchor=(1.02, 0.98),
          fontsize=11, frameon=True, framealpha=0.95)

# Layout
plt.tight_layout(rect=[0, 0, 0.78, 1])
fig.savefig('figure3b_quadrant_analysis.png', dpi=300, bbox_inches='tight')
plt.show()