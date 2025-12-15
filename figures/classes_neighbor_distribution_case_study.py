import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import os

# 1. SETUP DATA (Updated with SPECIFIC Foreign Neighbor Names)
# I included the specific neighbor name in the 'Class' column so it shows up on the X-axis
data = pd.DataFrame({
    'Class': [
        'Class 6\n(Foreign: Class 8)', 'Class 6\n(Foreign: Class 8)', 
        'Class 1\n(Foreign: Class 4)', 'Class 1\n(Foreign: Class 4)', 
        'Class 3\n(Foreign: Class 4)', 'Class 3\n(Foreign: Class 4)'
    ],
    'Type': [
        'Self-class', 'Top foreign neighbor', 
        'Self-class', 'Top foreign neighbor', 
        'Self-class', 'Top foreign neighbor'
    ],
    'Value': [51.1, 40.0, 60.4, 29.0, 50.9, 24.3]
})

# 2. SET THEME
sns.set_theme(style="white", context="poster", rc={
    "axes.spines.right": False,
    "axes.spines.top": False,
    "axes.linewidth": 3,
    "patch.linewidth": 3,
    "font.weight": "bold",
    "axes.labelweight": "bold"
})

# 3. CREATE PLOT
fig, ax = plt.subplots(figsize=(12, 7))

sns.barplot(
    data=data, 
    x='Class', 
    y='Value', 
    hue='Type', 
    palette=['#2ecc71', '#e74c3c'], 
    edgecolor='black',
    width=0.6,
    ax=ax
)

# 4. ADD LABELS ON BARS
for container in ax.containers:
    # Padding=4 keeps it tighter to the bar
    ax.bar_label(container, fmt='%.1f%%', padding=4, fontweight='bold', fontsize=24)

# 5. ADJUSTMENTS (Fixed the "Zooming too much")
# Reduced ylim from 90 -> 75. This makes the bars fill the chart better.
ax.set_ylim(0, 75) 
ax.set_xlabel("")
ax.set_ylabel("% of neighbors", fontsize=24, fontweight='bold')

# Legend: Moved inside to save space, but kept clean
ax.legend(loc='upper right', frameon=True, framealpha=1, edgecolor='black', fontsize=18)

plt.tight_layout()
os.makedirs("figures", exist_ok=True)
plt.savefig("figures/bar_chart_seaborn_fixed_zoom_labels.png", dpi=300)
plt.show()