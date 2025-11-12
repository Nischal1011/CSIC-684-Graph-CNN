import torch
import numpy as np
from torch_geometric.datasets import Amazon
from torch_geometric.transforms import NormalizeFeatures
import matplotlib.pyplot as plt
from torch_geometric.utils import degree
from torch_geometric.utils import to_networkx
import networkx as nx
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
# Load the Amazon Computers dataset
dataset = Amazon(root="/tmp/Amazon", name="Computers", transform=NormalizeFeatures())

# Access graph
data = dataset[0]

print(dataset)
print(data)
print("Number of nodes:", data.num_nodes)
print("Number of edges:", data.num_edges)
print("Number of features:", data.num_node_features)
print("Number of classes:", dataset.num_classes)

import collections
counts = collections.Counter(data.y.tolist())
print("Class counts:", counts)

for name in ["train_mask","val_mask","test_mask"]:
    m = getattr(data, name, None)
    if m is not None:
        print(f"{name}: {int(m.sum())} nodes ({float(m.sum())/data.num_nodes:.2%})")


# What: counts & proportions per class
# Why: choose metrics (macro-F1), consider class-weighted loss
counts = collections.Counter(data.y.tolist())
total = sum(counts.values())
props = {k: v/total for k,v in sorted(counts.items())}
print("Class counts:", dict(sorted(counts.items())))
print("Class proportions:", props)

plt.figure(figsize=(6,3))
plt.bar(list(props.keys()), list(props.values()))
plt.xticks(range(dataset.num_classes))
plt.ylabel("Proportion"); plt.title("Class distribution"); plt.tight_layout(); plt.show()


# What: how sparse BOW features are
# Why: guide normalization/regularization and PCA/TSNE speed
X = data.x
zero_frac = float((X==0).sum())/X.numel()
nonzero_per_node = (X!=0).sum(dim=1).cpu().numpy()
print(f"Sparsity (fraction of zeros): {zero_frac:.4f}")
print(f"Avg non-zero features per node: {nonzero_per_node.mean():.1f}")

plt.figure(figsize=(6,3))
plt.hist(nonzero_per_node, bins=50)
plt.xlabel("# non-zero features / node"); plt.ylabel("Count")
plt.title("Feature density per node"); plt.tight_layout(); plt.show()



# What: self-loops? multiple edges? undirected?
# Why: training stability and fair comparisons

# self-loops
ei = data.edge_index
self_loops = (ei[0] == ei[1]).sum().item()
print("Self-loops in stored edge_index:", self_loops)


from torch_geometric.utils import to_undirected
ei = data.edge_index

# manual undirected check
# sort each edge and see if its reverse exists
edges = ei.t().tolist()
edges_set = set(map(tuple, edges))
rev_edges_set = set((j,i) for (i,j) in edges)
is_undir = edges_set == rev_edges_set

edge_index_undir = ei if is_undir else to_undirected(ei)
print("Is undirected (stored)?", is_undir)
print("Edges (undirected view):", edge_index_undir.size(1))



# What: node degrees (mean/med/max, heavy tail)
# Why: hubs vs isolates affect message passing & sampling

deg = degree(edge_index_undir[0], num_nodes=data.num_nodes).cpu().numpy()
print(f"Mean degree: {deg.mean():.2f} | Median: {np.median(deg):.0f} | Max: {deg.max():.0f}")

plt.figure(figsize=(6,3))
plt.hist(deg, bins=80)
plt.xlabel("Degree"); plt.ylabel("Count"); plt.title("Degree distribution")
plt.tight_layout(); plt.show()

# quick log-log view
hist, bins = np.histogram(deg, bins=80)
mask = (hist>0) & (bins[:-1]>0)
plt.figure(figsize=(6,3))
plt.plot(np.log10(bins[:-1][mask]), np.log10(hist[mask]), 'o', ms=3, ls='none')
plt.xlabel("log10(degree)"); plt.ylabel("log10(freq)")
plt.title("Degree distribution (log-log)"); plt.tight_layout(); plt.show()


# What: number/size of connected components
# Why: informs limits of message passing; motivates community ideas
G = to_networkx(data, to_undirected=True)
comp_sizes = [len(c) for c in nx.connected_components(G)]
comp_sizes.sort(reverse=True)
print("Num components:", len(comp_sizes))
print("Largest component:", comp_sizes[0], "| Smallest:", comp_sizes[-1])
print("Top-10 component sizes:", comp_sizes[:10])

plt.figure(figsize=(6,3))
plt.hist(comp_sizes, bins=50)
plt.xlabel("Component size"); plt.ylabel("Count")
plt.title("Connected component sizes"); plt.tight_layout(); plt.show()


# What: fraction of edges connecting same-label nodes
# Why: gauges how helpful graph structure will be for GCN

y = data.y
same = (y[edge_index_undir[0]] == y[edge_index_undir[1]]).sum().item()
homophily = same / edge_index_undir.size(1)
print(f"Edge homophily (same-label fraction): {homophily:.4f}")


# What: visualize separability by features alone
# Why: baseline intuition; later compare after GCN

X_np = X.cpu().numpy(); y_np = y.cpu().numpy()

# PCA to 50D then t-SNE to 2D (subsample for speed)
pca_50 = PCA(n_components=min(50, X_np.shape[1]), random_state=42).fit_transform(X_np)
idx = np.random.RandomState(42).choice(len(pca_50), size=min(8000, len(pca_50)), replace=False)
pca_50 = pca_50[idx]; y_sub = y_np[idx]

emb2 = TSNE(n_components=2, perplexity=30, init='pca', learning_rate='auto', random_state=42).fit_transform(pca_50)

plt.figure(figsize=(7,5))
sc = plt.scatter(emb2[:,0], emb2[:,1], c=y_sub, cmap="tab20", s=5)
plt.colorbar(sc, label="Class")
plt.title("t-SNE of node FEATURES (no graph)"); plt.tight_layout(); plt.show()


# What: train/val/test label balance & dispersion
# Why: avoid skewed splits; know what you’re training on

if getattr(data, "train_mask", None) is not None:
    train_counts = collections.Counter(y[data.train_mask].tolist())
    val_counts   = collections.Counter(y[data.val_mask].tolist())
    test_counts  = collections.Counter(y[data.test_mask].tolist())
    print("Train class counts:", dict(sorted(train_counts.items())))
    print("Val   class counts:", dict(sorted(val_counts.items())))
    # test is huge; show top few
    print("Test class counts (top):", dict(sorted(test_counts.items()) ) )


# What: degree & component id as extra features
# Why: let LR/SVM get a pinch of “structure” too (fairer baseline)

# degree already computed: deg
# component id:
comp_id_map = {}
for cid, comp in enumerate(nx.connected_components(G)):
    for n in comp:
        comp_id_map[n] = cid
comp_id = np.array([comp_id_map[i] for i in range(data.num_nodes)])

print("Num components:", comp_id.max()+1, "| Example comp ids:", np.unique(comp_id)[:10])


# What: print once; paste into report
# Why: ties EDA to your story
print("\n=== QUICK SUMMARY ===")
print(f"Nodes={data.num_nodes:,}, Edges={data.num_edges:,}, Feats={data.num_node_features}, Classes={dataset.num_classes}")
print(f"Sparsity={zero_frac:.3f}, MeanDeg={deg.mean():.2f}, Homophily={homophily:.3f}")
print(f"Components={len(comp_sizes)} | Largest={comp_sizes[0]} | Smallest={comp_sizes[-1]}")
print("Top class counts:", dict(collections.Counter(y.tolist()).most_common(5)))





# 425 - labeled - ground truth (selected randomly)
# semi supervised GCN (425  + unlableled nodes for training) -> 

# fully labeled training for logistic, SVM, GCN, semi GCN (randomly remove hypter parameter - [40%-60%]), GCN - community signals (optional), semi GCN - community signals [40%-60%]


# 0. 1000 
# 1. 1000 
# 2  1000  (x - 1000)
# 3  1000 (x - 1000)

# disconnected components - rid of this (before we do selection)


# data properties - Homophily=0.777, dependency (not iid (classical ML)) 



# 695 - unlabeled - to predict (randomly selected)  (label) test set 


# sunday meeting (pitch) (1 pm?) (deck)
# EDA - to put slide - whole data properties (no iid) hopmopy; graph; hypohesis ; how we massage the data to fit out hyposis (computational efficieny and also for testing semi-supervised GCN to prevent bais random selection , class distribution), evaluation baseline (logic-) class concepts, final 4 models, logistic, SVM, GCN, GCN with some unlabled (40-60) (bias-variance) - report the best gene; evaluation metric - f1, ROC, AUC
# random subset of the data 
# fit classical ML model (test data) results - 4 classes 