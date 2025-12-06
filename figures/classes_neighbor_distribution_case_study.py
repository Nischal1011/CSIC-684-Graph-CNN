import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch

# --------------------------------
# Data for the three low-homophily classes
# --------------------------------
class_data = {
    'Class 6': {
        'self': 51.1,
        'foreign': 40.0,
        'foreign_label': 'Class 8',
        'homophily': 0.511
    },
    'Class 1': {
        'self': 60.4,
        'foreign': 29.0,
        'foreign_label': 'Class 4',
        'homophily': 0.604
    },
    'Class 3': {
        'self': 50.9,
        'foreign': 24.3,
        'foreign_label': 'Class 4',
        'homophily': 0.509
    }
}

# --------------------------------
# Figure setup
# --------------------------------
fig, axes = plt.subplots(1, 3, figsize=(15, 6))
bar_width = 0.6

for idx, (class_name, data) in enumerate(class_data.items()):
    ax = axes[idx]
    
    # Bars
    categories = [f'{class_name}\n(Self)', f'{data["foreign_label"]}\n(Foreign)']
    values = [data['self'], data['foreign']]
    colors = ['#2ecc71', '#e74c3c']
    
    bars = ax.bar(categories, values, width=bar_width,
                  color=colors, edgecolor='black', linewidth=2)

    # Value labels
    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width()/2., val + 1.2,
                f'{val:.1f}%', ha='center', va='bottom',
                fontsize=14, fontweight='bold')

    # Axes styling
    ax.set_ylim(0, 70)
    ax.set_ylabel('% of Neighbors', fontsize=11, fontweight='bold')
    ax.set_title(f'{class_name} Neighbor Distribution', fontsize=13, fontweight='bold')
    ax.yaxis.grid(True, linestyle='--', alpha=0.5)
    ax.set_axisbelow(True)

    # Homophily annotation
    ax.text(0.5, 0.92, f'Homophily: {data["homophily"]:.1%}',
            transform=ax.transAxes, ha='center',
            fontsize=11, fontstyle='italic',
            bbox=dict(boxstyle='round,pad=0.3',
                      facecolor='lightyellow', alpha=0.8))

    # Highlight Class 6 with extra annotation
    if class_name == 'Class 6':
        ax.text(0.5, 63.5,
                "Near 50–50 split\nGCN gets mixed signals",
                ha='center', va='bottom',
                fontsize=10, fontweight='bold', color='darkred',
                bbox=dict(boxstyle='round,pad=0.4',
                          facecolor='white', edgecolor='darkred', linewidth=1.2))

# --------------------------------
# Legend inside figure
# --------------------------------
legend_elements = [
    Patch(facecolor='#2ecc71', edgecolor='black', label='Self-class (correct signal)'),
    Patch(facecolor='#e74c3c', edgecolor='black', label='Top foreign neighbor (noise)')
]
fig.legend(handles=legend_elements, loc='upper center', ncol=2,
           fontsize=11, bbox_to_anchor=(0.5, 0.07), frameon=True)

# --------------------------------
# Main title
# --------------------------------
fig.suptitle(
    'Neighbor Distribution for Low-Homophily Classes\n'
    '(Dominant foreign neighbors introduce confusion for GCN)',
    fontsize=15, fontweight='bold', y=0.98
)

# --------------------------------
# Adjust spacing
# --------------------------------
plt.subplots_adjust(top=0.88, bottom=0.17, wspace=0.25)

# --------------------------------
# Save + show
# --------------------------------
plt.savefig('figure4_neighbor_distribution_clean.png', dpi=300, bbox_inches='tight')
plt.show()
