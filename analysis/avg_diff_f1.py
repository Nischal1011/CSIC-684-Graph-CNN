import numpy as np
from scipy.stats import pearsonr, spearmanr

# Per-class F1 scores from your results
gcn_f1 = {
    '10%': {0: 0.896, 1: 0.861, 2: 0.940, 3: 0.815, 4: 0.878, 
            5: 0.938, 6: 0.576, 7: 0.881, 8: 0.742, 9: 0.840},
    '50%': {0: 0.896, 1: 0.856, 2: 0.948, 3: 0.835, 4: 0.874, 
            5: 0.951, 6: 0.590, 7: 0.908, 8: 0.755, 9: 0.873},
    '100%': {0: 0.893, 1: 0.865, 2: 0.955, 3: 0.837, 4: 0.879, 
             5: 0.946, 6: 0.594, 7: 0.916, 8: 0.761, 9: 0.849}
}

lr_f1 = {
    '10%': {0: 0.661, 1: 0.830, 2: 0.877, 3: 0.849, 4: 0.877, 
            5: 0.830, 6: 0.455, 7: 0.849, 8: 0.764, 9: 0.754},
    '50%': {0: 0.770, 1: 0.865, 2: 0.916, 3: 0.884, 4: 0.887, 
            5: 0.918, 6: 0.545, 7: 0.886, 8: 0.754, 9: 0.874},
    '100%': {0: 0.764, 1: 0.865, 2: 0.914, 3: 0.913, 4: 0.881, 
             5: 0.916, 6: 0.565, 7: 0.879, 8: 0.749, 9: 0.878}
}

# Homophily values
class_homophily = {
    0: 0.6689, 1: 0.6036, 2: 0.8879, 3: 0.5088, 4: 0.8587,
    5: 0.9147, 6: 0.5106, 7: 0.9051, 8: 0.6645, 9: 0.7788
}

# Compute average ΔF1 across label rates
classes = list(range(10))
delta_f1_avg = {}

for c in classes:
    delta_10 = gcn_f1['10%'][c] - lr_f1['10%'][c]
    delta_50 = gcn_f1['50%'][c] - lr_f1['50%'][c]
    delta_100 = gcn_f1['100%'][c] - lr_f1['100%'][c]
    delta_f1_avg[c] = (delta_10 + delta_50 + delta_100) / 3

print("=" * 60)
print("AVERAGE ΔF1(GCN - LR) ACROSS ALL LABEL RATES")
print("=" * 60)
for c in classes:
    print(f"  Class {c}: ΔF1 = {delta_f1_avg[c]:+.4f}, Homophily = {class_homophily[c]:.4f}")

# Compute correlation
homophily_arr = np.array([class_homophily[c] for c in classes])
delta_f1_arr = np.array([delta_f1_avg[c] for c in classes])

pearson_r, pearson_p = pearsonr(homophily_arr, delta_f1_arr)
spearman_r, spearman_p = spearmanr(homophily_arr, delta_f1_arr)

print(f"\n" + "=" * 60)
print("CORRELATION: HOMOPHILY vs GCN GAIN")
print("=" * 60)
print(f"  Pearson r:  {pearson_r:.4f} (p = {pearson_p:.4f})")
print(f"  Spearman r: {spearman_r:.4f} (p = {spearman_p:.4f})")

if pearson_p < 0.05:
    print(f"\n  Result: SIGNIFICANT positive correlation")
    print(f"  Interpretation: Homophily predicts GCN benefit")
else:
    print(f"\n  Result: NOT statistically significant (p > 0.05)")
    print(f"  Interpretation: Need to investigate further")

# Show the pattern clearly
print(f"\n" + "=" * 60)
print("PATTERN CHECK")
print("=" * 60)
print("\nLow homophily classes (<0.55):")
for c in classes:
    if class_homophily[c] < 0.55:
        print(f"  Class {c}: homophily={class_homophily[c]:.3f}, ΔF1={delta_f1_avg[c]:+.3f}")

print("\nHigh homophily classes (>0.85):")
for c in classes:
    if class_homophily[c] > 0.85:
        print(f"  Class {c}: homophily={class_homophily[c]:.3f}, ΔF1={delta_f1_avg[c]:+.3f}")