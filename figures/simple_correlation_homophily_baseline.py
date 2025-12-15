import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
import os

try:
    from scipy import stats
except Exception as e:
    raise ImportError("This script requires scipy. Install it with `pip install scipy` and re-run.") from e

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

# Journal palette
PALETTE_POS = '#4C72B0'   # GCN helps
PALETTE_NEG = '#DD8452'   # GCN hurts
FIT_COLOR = '#2F2F2F'

colors = [PALETTE_POS if d > 0 else PALETTE_NEG for d in delta_f1]

# -----------------------------
# Helpers: linear fit + annotate p-value
# -----------------------------
def p_to_stars(p):
    if p < 0.001: return '***'
    if p < 0.01: return '**'
    if p < 0.05: return '*'
    return ''

def fit_and_annotate(ax, x, y, xpos, ypos, fit_color=FIT_COLOR):
    coeffs = np.polyfit(x, y, 1)
    slope, intercept = coeffs[0], coeffs[1]
    r, pval = stats.pearsonr(x, y)
    x_line = np.linspace(x.min() - 0.02, x.max() + 0.02, 200)
    y_line = np.polyval(coeffs, x_line)
    ax.plot(x_line, y_line, linestyle='-', linewidth=1.6, color=fit_color, zorder=1)
    
    if pval < 0.001: p_text = "p < 0.001"
    else: p_text = f"p = {pval:.3f}"
    stars = p_to_stars(pval)
    
    ax.text(xpos, ypos,
            f"r = {r:.2f} {stars}\nslope = {slope:.3f}\n{p_text}",
            transform=ax.transAxes, fontsize=10, fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.92),
            zorder=5)
    return r, pval, slope, intercept

# -----------------------------
# Figure setup
# -----------------------------
plt.rcParams.update({
    "font.size": 12,
    "axes.spines.top": False,
    "axes.spines.right": False,
})

FIG_W, FIG_H = 8, 10  # taller for vertical stack
fig, axes = plt.subplots(2, 1, figsize=(FIG_W, FIG_H))
plt.subplots_adjust(hspace=0.35)

def beautify_axes(ax):
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.tick_params(axis='both', labelsize=11)
    ax.grid(True, linestyle='--', alpha=0.28)

# -----------------------------
# Plot 1: Homophily vs ΔF1
# -----------------------------
ax1 = axes[0]
ax1.scatter(homophily, delta_f1, c=colors, s=220, edgecolor='black', linewidth=1.1, zorder=3)
ax1.axhline(y=0, color='gray', linestyle='--', linewidth=1.1, alpha=0.7)
ax1.set_xlabel('Homophily', fontsize=12, fontweight='bold')
ax1.set_ylabel('ΔF1 (GCN - LR)', fontsize=12, fontweight='bold')
ax1.set_xlim(0.45, 0.98)
ax1.set_ylim(-0.10, 0.20)
r1, p1, s1, _ = fit_and_annotate(ax1, homophily, delta_f1, xpos=0.65, ypos=0.10, fit_color=FIT_COLOR)
beautify_axes(ax1)

# -----------------------------
# Plot 2: LR Baseline vs ΔF1
# -----------------------------
ax2 = axes[1]
ax2.scatter(lr_baseline, delta_f1, c=colors, s=220, edgecolor='black', linewidth=1.1, zorder=3)
ax2.axhline(y=0, color='gray', linestyle='--', linewidth=1.1, alpha=0.7)
ax2.set_xlabel('LR Baseline F1 (Avg)', fontsize=12, fontweight='bold')
ax2.set_ylabel('ΔF1 (GCN - LR)', fontsize=12, fontweight='bold')
ax2.set_xlim(0.48, 0.95)
ax2.set_ylim(-0.10, 0.20)
r2, p2, s2, _ = fit_and_annotate(ax2, lr_baseline, delta_f1, xpos=0.65, ypos=0.10, fit_color=FIT_COLOR)
beautify_axes(ax2)

# -----------------------------
# Legend (top right)
# -----------------------------
legend_elements = [
    Line2D([0], [0], marker='o', color='w', markerfacecolor=PALETTE_POS,
           markersize=10, markeredgecolor='black', label='GCN helps (ΔF1 > 0)'),
    Line2D([0], [0], marker='o', color='w', markerfacecolor=PALETTE_NEG,
           markersize=10, markeredgecolor='black', label='GCN hurts (ΔF1 < 0)'),
    Line2D([0], [0], linestyle='-', color=FIT_COLOR, linewidth=1.6, label='Linear fit')
]
ax2.legend(handles=legend_elements, loc='upper right', fontsize=11, frameon=True)

# -----------------------------
# Save figure
# -----------------------------
os.makedirs("figures", exist_ok=True)
fig.savefig('figures/figure3_vertical_clean.png', dpi=600, bbox_inches='tight')
print("✓ Saved: figures/figure3_vertical_clean.png")

plt.show()

# -----------------------------
# Print statistics
# -----------------------------
def fmt_p(p): return "<0.001" if p < 0.001 else f"{p:.3f}"
print(f'Homophily vs ΔF1: Pearson r = {r1:.3f}, p = {fmt_p(p1)}, slope = {s1:.4f}')
print(f'LR baseline vs ΔF1: Pearson r = {r2:.3f}, p = {fmt_p(p2)}, slope = {s2:.4f}')
