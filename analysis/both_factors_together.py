import numpy as np

# Data
classes = list(range(10))

lr_f1_avg = {
    0: 0.732, 1: 0.853, 2: 0.902, 3: 0.882, 4: 0.882,
    5: 0.888, 6: 0.522, 7: 0.871, 8: 0.756, 9: 0.835
}

delta_f1_avg = {
    0: 0.1633, 1: 0.0073, 2: 0.0453, 3: -0.0530, 4: -0.0047,
    5: 0.0570, 6: 0.0650, 7: 0.0303, 8: -0.0030, 9: 0.0187
}

class_homophily = {
    0: 0.6689, 1: 0.6036, 2: 0.8879, 3: 0.5088, 4: 0.8587,
    5: 0.9147, 6: 0.5106, 7: 0.9051, 8: 0.6645, 9: 0.7788
}

print("=" * 70)
print("QUADRANT ANALYSIS: Homophily × Baseline Difficulty")
print("=" * 70)

# Define thresholds
homo_threshold = 0.70  # median split
baseline_threshold = 0.85  # median split

print(f"\nThresholds: Homophily={homo_threshold}, LR Baseline={baseline_threshold}")
print()

# Categorize classes
quadrants = {
    "Low Homo, Hard Baseline": [],
    "Low Homo, Easy Baseline": [],
    "High Homo, Hard Baseline": [],
    "High Homo, Easy Baseline": []
}

for c in classes:
    homo = class_homophily[c]
    baseline = lr_f1_avg[c]
    delta = delta_f1_avg[c]
    
    if homo < homo_threshold and baseline < baseline_threshold:
        quadrants["Low Homo, Hard Baseline"].append((c, delta, homo, baseline))
    elif homo < homo_threshold and baseline >= baseline_threshold:
        quadrants["Low Homo, Easy Baseline"].append((c, delta, homo, baseline))
    elif homo >= homo_threshold and baseline < baseline_threshold:
        quadrants["High Homo, Hard Baseline"].append((c, delta, homo, baseline))
    else:
        quadrants["High Homo, Easy Baseline"].append((c, delta, homo, baseline))

for quadrant, members in quadrants.items():
    print(f"\n{quadrant}:")
    if not members:
        print("  (none)")
    else:
        avg_delta = np.mean([m[1] for m in members])
        for c, delta, homo, baseline in members:
            print(f"  Class {c}: ΔF1={delta:+.3f}, Homo={homo:.3f}, LR={baseline:.3f}")
        print(f"  --> Average ΔF1: {avg_delta:+.4f}")

print()
print("=" * 70)
print("INTERPRETATION")
print("=" * 70)