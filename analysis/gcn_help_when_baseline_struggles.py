import numpy as np
from scipy.stats import pearsonr, spearmanr

# Average LR F1 across label rates (how well does baseline do?)
lr_f1_avg = {
    0: (0.661 + 0.770 + 0.764) / 3,  # 0.732
    1: (0.830 + 0.865 + 0.865) / 3,  # 0.853
    2: (0.877 + 0.916 + 0.914) / 3,  # 0.902
    3: (0.849 + 0.884 + 0.913) / 3,  # 0.882
    4: (0.877 + 0.887 + 0.881) / 3,  # 0.882
    5: (0.830 + 0.918 + 0.916) / 3,  # 0.888
    6: (0.455 + 0.545 + 0.565) / 3,  # 0.522
    7: (0.849 + 0.886 + 0.879) / 3,  # 0.871
    8: (0.764 + 0.754 + 0.749) / 3,  # 0.756
    9: (0.754 + 0.874 + 0.878) / 3,  # 0.835
}

delta_f1_avg = {
    0: 0.1633, 1: 0.0073, 2: 0.0453, 3: -0.0530, 4: -0.0047,
    5: 0.0570, 6: 0.0650, 7: 0.0303, 8: -0.0030, 9: 0.0187
}

class_homophily = {
    0: 0.6689, 1: 0.6036, 2: 0.8879, 3: 0.5088, 4: 0.8587,
    5: 0.9147, 6: 0.5106, 7: 0.9051, 8: 0.6645, 9: 0.7788
}

classes = list(range(10))

# Arrays for correlation
lr_baseline = np.array([lr_f1_avg[c] for c in classes])
delta_f1 = np.array([delta_f1_avg[c] for c in classes])
homophily = np.array([class_homophily[c] for c in classes])

print("=" * 60)
print("HYPOTHESIS: GCN HELPS WHEN BASELINE STRUGGLES")
print("=" * 60)

# Correlation: Baseline performance vs GCN gain
pearson_r, pearson_p = pearsonr(lr_baseline, delta_f1)
spearman_r, spearman_p = spearmanr(lr_baseline, delta_f1)

print(f"\nCorrelation: LR Baseline F1 vs ΔF1(GCN-LR)")
print(f"  Pearson r:  {pearson_r:.4f} (p = {pearson_p:.4f})")
print(f"  Spearman r: {spearman_r:.4f} (p = {spearman_p:.4f})")

print(f"\n" + "-" * 60)
print("PER-CLASS BREAKDOWN")
print("-" * 60)
print(f"{'Class':<8} {'LR Baseline':<12} {'ΔF1':<10} {'Homophily':<10}")
print("-" * 60)
for c in classes:
    print(f"{c:<8} {lr_f1_avg[c]:<12.3f} {delta_f1_avg[c]:+<10.3f} {class_homophily[c]:<10.3f}")

# Check: do the two worst baseline classes get the most help?
print(f"\n" + "=" * 60)
print("KEY OBSERVATION")
print("=" * 60)
sorted_by_baseline = sorted(classes, key=lambda c: lr_f1_avg[c])
print("\nClasses sorted by baseline difficulty (worst first):")
for c in sorted_by_baseline:
    print(f"  Class {c}: LR={lr_f1_avg[c]:.3f}, ΔF1={delta_f1_avg[c]:+.3f}, Homophily={class_homophily[c]:.3f}")