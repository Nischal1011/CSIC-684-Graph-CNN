import torch
from collections import Counter
import numpy as np

# Load data
data = torch.load('data/data_with_split.pt', weights_only=False)
edge_index = data.edge_index

def get_neighbor_distribution(data, class_id):
    """Get the neighbor class distribution for nodes of a given class."""
    class_nodes = (data.y == class_id).nonzero(as_tuple=True)[0]
    
    neighbor_labels = []
    for node in class_nodes:
        neighbors = edge_index[1][edge_index[0] == node]
        neighbor_labels.extend(data.y[neighbors].tolist())
    
    neighbor_counts = Counter(neighbor_labels)
    total = sum(neighbor_counts.values())
    
    distribution = {}
    for cls in range(10):
        count = neighbor_counts.get(cls, 0)
        distribution[cls] = count / total * 100 if total > 0 else 0
    
    return distribution

# Compute neighbor distribution for all classes
print("=" * 70)
print("NEIGHBOR DISTRIBUTION FOR ALL CLASSES")
print("=" * 70)

all_distributions = {}
for c in range(10):
    all_distributions[c] = get_neighbor_distribution(data, c)

# Print as a matrix
print("\n[Rows = Source Class, Columns = Neighbor Class]")
print()
header = "Source |" + "".join([f"  N={i}  " for i in range(10)])
print(header)
print("-" * len(header))

for src in range(10):
    row = f"  C{src}   |"
    for dst in range(10):
        pct = all_distributions[src][dst]
        if src == dst:
            row += f" [{pct:4.1f}%]"  # Highlight self-loops (homophily)
        elif pct >= 15:
            row += f" *{pct:4.1f}%"  # Highlight high foreign neighbors
        else:
            row += f"  {pct:4.1f}%"
    print(row)

print()
print("Legend: [XX.X%] = self-class (homophily), *XX.X% = high foreign (≥15%)")

# Find top confusion pairs (high foreign neighbor %)
print()
print("=" * 70)
print("TOP FOREIGN NEIGHBOR PAIRS (≥15%)")
print("=" * 70)

confusion_pairs = []
for src in range(10):
    for dst in range(10):
        if src != dst and all_distributions[src][dst] >= 15:
            confusion_pairs.append((src, dst, all_distributions[src][dst]))

confusion_pairs.sort(key=lambda x: -x[2])

print(f"\n{'Source':<10} {'Neighbor':<10} {'Percentage':<12} Interpretation")
print("-" * 50)
for src, dst, pct in confusion_pairs:
    print(f"Class {src:<4} Class {dst:<4} {pct:5.1f}%        C{src}→C{dst} confusion likely")

# Now let's check if these match actual confusion in predictions
print()
print("=" * 70)
print("HOMOPHILY SUMMARY BY CLASS")
print("=" * 70)
print(f"\n{'Class':<8} {'Homophily':<12} {'Top Foreign':<15} {'Foreign %':<12}")
print("-" * 50)

for c in range(10):
    homophily = all_distributions[c][c]
    # Find top foreign neighbor
    foreign = [(dst, pct) for dst, pct in all_distributions[c].items() if dst != c]
    top_foreign = max(foreign, key=lambda x: x[1])
    print(f"Class {c:<3} {homophily:5.1f}%       Class {top_foreign[0]:<10} {top_foreign[1]:5.1f}%")