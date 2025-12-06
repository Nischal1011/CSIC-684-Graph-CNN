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
colors = ['#2ecc71' if d > 0 else '#e74c3c' for d in delta_f1]

# --------------------------------
# Helper: linear fit + annotate
# --------------------------------
def fit_and_annotate(ax, x, y, xpos, ypos):
    coeffs = np.polyfit(x, y, 1)
    slope, intercept = coeffs[0], coeffs[1]
    r = np.corrcoef(x, y)[0,1]
    x_line = np.linspace(x.min()-0.02, x.max()+0.02, 200)
    y_line = np.polyval(coeffs, x_line)
    ax.plot(x_line, y_line, linestyle='-', linewidth=1.5, label='Linear fit', zorder=1)
    ax.text(xpos, ypos, f'r = {r:.2f}\nslope = {slope:.3f}',
            transform=ax.transAxes, fontsize=10, fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.85))
    return r, slope, intercept

# --------------------------------
# Figure
# --------------------------------
plt.rcParams.update({
    "font.size": 12,
    "axes.spines.top": False,
    "axes.spines.right": False,
})

# INCREASED HEIGHT from (14,5) to (14,6) for better spacing
fig, axes = plt.subplots(1, 2, figsize=(14,6))

# Main title - LOWERED y position from 1.02 to 0.98
fig.suptitle("GCN Benefit vs Homophily and Feature Strength (Avg ΔF1)", 
             fontsize=16, fontweight='bold', y=0.98)

# -----------------------------
# Plot 1: Homophily vs ΔF1
# -----------------------------
ax1 = axes[0]
for i, c in enumerate(classes):
    ax1.scatter(homophily[i], delta_f1[i], c=colors[i], s=200,
                edgecolor='black', linewidth=1.2, zorder=3)
    ax1.annotate(f'C{c}', (homophily[i], delta_f1[i]),
                 textcoords="offset points", xytext=(8,0),
                 ha='left', fontsize=10, fontweight='bold')
ax1.axhline(y=0, color='gray', linestyle='--', linewidth=1.2, alpha=0.7)
ax1.set_xlabel('Homophily', fontsize=12, fontweight='bold')
ax1.set_ylabel('ΔF1 (GCN - LR)', fontsize=12, fontweight='bold')
ax1.set_title('Homophily vs ΔF1', fontsize=13, fontweight='bold')
ax1.set_xlim(0.45,0.98)
ax1.set_ylim(-0.10,0.20)
ax1.yaxis.grid(True, linestyle='--', alpha=0.5)
ax1.xaxis.grid(True, linestyle='--', alpha=0.5)
ax1.tick_params(axis='x', labelrotation=30)
ax1.xaxis.labelpad = 10
r1,s1,_ = fit_and_annotate(ax1, homophily, delta_f1, xpos=0.55, ypos=0.10)
ax1.text(0.5,0.95,'Scattered points — no strong linear pattern',
         transform=ax1.transAxes, fontsize=10, fontstyle='italic', color='gray',
         ha='center', va='top', bbox=dict(boxstyle='round,pad=0.3', facecolor='lightyellow', alpha=0.9))

# -----------------------------
# Plot 2: LR Baseline vs ΔF1
# -----------------------------
ax2 = axes[1]
for i,c in enumerate(classes):
    ax2.scatter(lr_baseline[i], delta_f1[i], c=colors[i], s=200,
                edgecolor='black', linewidth=1.2, zorder=3)
    ax2.annotate(f'C{c}', (lr_baseline[i], delta_f1[i]),
                 textcoords="offset points", xytext=(8,0),
                 ha='left', fontsize=10, fontweight='bold')
ax2.axhline(y=0, color='gray', linestyle='--', linewidth=1.2, alpha=0.7)
ax2.set_xlabel('LR Baseline F1 (Avg)', fontsize=12, fontweight='bold')
ax2.set_ylabel('ΔF1 (GCN - LR)', fontsize=12, fontweight='bold')
ax2.set_title('LR Baseline vs ΔF1', fontsize=13, fontweight='bold')
ax2.set_xlim(0.48,0.95)
ax2.set_ylim(-0.10,0.20)
ax2.yaxis.grid(True, linestyle='--', alpha=0.5)
ax2.xaxis.grid(True, linestyle='--', alpha=0.5)
ax2.tick_params(axis='x', labelrotation=30)
ax2.xaxis.labelpad = 10
r2,s2,_ = fit_and_annotate(ax2, lr_baseline, delta_f1, xpos=0.55, ypos=0.10)
ax2.text(0.5,0.95,'Scattered points — no monotonic relationship',
         transform=ax2.transAxes, fontsize=10, fontstyle='italic', color='gray',
         ha='center', va='top', bbox=dict(boxstyle='round,pad=0.3', facecolor='lightyellow', alpha=0.9))

# -----------------------------
# Legend (moved up and inside figure area)
# -----------------------------
legend_elements = [
    Line2D([0],[0], marker='o', color='w', markerfacecolor='#2ecc71',
           markersize=10, markeredgecolor='black', label='GCN helps (ΔF1 > 0)'),
    Line2D([0],[0], marker='o', color='w', markerfacecolor='#e74c3c',
           markersize=10, markeredgecolor='black', label='GCN hurts (ΔF1 < 0)'),
    Line2D([0],[0], linestyle='-', color='black', label='Linear fit')
]

# RAISED bbox_to_anchor y from 0.02 to 0.12 to place legend higher
fig.legend(handles=legend_elements, loc='upper center', ncol=3, fontsize=11,
           bbox_to_anchor=(0.5, 0.12), frameon=True)

# -----------------------------
# Adjust layout
# -----------------------------
# INCREASED top margin from 0.88 to 0.85 (lower number = more top space)
# DECREASED bottom margin from 0.18 to 0.12 (higher number = less bottom space)
plt.subplots_adjust(bottom=0.12, top=0.85, wspace=0.3)

# ADD tight_layout to prevent clipping
plt.tight_layout(rect=[0, 0.05, 1, 0.93])  # rect=[left, bottom, right, top]

# -----------------------------
# Save + show
# -----------------------------
fig.savefig('figure3a_final_with_title_legend.png', dpi=300, bbox_inches='tight')
plt.show()

print(f'Homophily vs ΔF1: Pearson r = {r1:.3f}, slope = {s1:.4f}')
print(f'LR baseline vs ΔF1: Pearson r = {r2:.3f}, slope = {s2:.4f}')