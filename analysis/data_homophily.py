# The key property is homophily - the tendency of connected nodes to share the same label. When homophily is high, a GCN can essentially "vote" based on neighbors' labels, making node features almost irrelevant.

import torch
import numpy as np
from torch_geometric.datasets import Amazon
from collections import Counter

def compute_edge_homophily(data):
    """
    Edge homophily: fraction of edges connecting nodes of the same class.
    Range: [0, 1], where 1 = perfect homophily
    """
    edge_index = data.edge_index
    src_labels = data.y[edge_index[0]]
    dst_labels = data.y[edge_index[1]]
    same_label = (src_labels == dst_labels).float()
    return same_label.mean().item()

def compute_node_homophily(data):
    """
    Node homophily: average fraction of neighbors with the same label as the node.
    """
    edge_index = data.edge_index
    num_nodes = data.num_nodes
    
    homophily_scores = []
    for node in range(num_nodes):
        # Find neighbors
        neighbors = edge_index[1][edge_index[0] == node]
        if len(neighbors) == 0:
            continue
        neighbor_labels = data.y[neighbors]
        node_label = data.y[node]
        same_class_ratio = (neighbor_labels == node_label).float().mean().item()
        homophily_scores.append(same_class_ratio)
    
    return np.mean(homophily_scores)

def compute_class_homophily(data):
    """
    Per-class homophily: for each class, what fraction of edges stay within class?
    """
    edge_index = data.edge_index
    num_classes = data.y.max().item() + 1
    
    class_homophily = {}
    for c in range(num_classes):
        # Find nodes of this class
        class_nodes = (data.y == c).nonzero(as_tuple=True)[0]
        class_node_set = set(class_nodes.tolist())
        
        # Find edges originating from this class
        src_mask = torch.isin(edge_index[0], class_nodes)
        src_of_class = edge_index[0][src_mask]
        dst_of_class = edge_index[1][src_mask]
        
        # Count how many go to same class
        dst_labels = data.y[dst_of_class]
        same_class = (dst_labels == c).float().mean().item()
        class_homophily[c] = same_class
    
    return class_homophily

def compute_label_propagation_accuracy(data, num_iterations=2):
    """
    Simulate what GCN effectively does: propagate labels through the graph.
    This shows the theoretical ceiling for structure-only classification.
    """
    edge_index = data.edge_index
    num_nodes = data.num_nodes
    num_classes = data.y.max().item() + 1
    
    # Initialize with one-hot labels
    label_dist = torch.zeros(num_nodes, num_classes)
    label_dist[torch.arange(num_nodes), data.y] = 1.0
    
    # Build adjacency (normalized)
    from torch_geometric.utils import degree
    row, col = edge_index
    deg = degree(col, num_nodes)
    deg_inv_sqrt = deg.pow(-0.5)
    deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0
    
    # Propagate
    for _ in range(num_iterations):
        # Aggregate neighbor labels
        new_dist = torch.zeros_like(label_dist)
        for i in range(edge_index.size(1)):
            src, dst = edge_index[0, i].item(), edge_index[1, i].item()
            weight = deg_inv_sqrt[src] * deg_inv_sqrt[dst]
            new_dist[dst] += weight * label_dist[src]
        label_dist = new_dist
    
    # Predict based on aggregated labels
    predictions = label_dist.argmax(dim=1)
    accuracy = (predictions == data.y).float().mean().item()
    return accuracy

if __name__ == "__main__":
    # Load data
    data = torch.load('data/data_with_split.pt', weights_only=False)
    
    print("=" * 60)
    print("GRAPH STRUCTURE ANALYSIS: Why Random Embeddings Work")
    print("=" * 60)
    
    # 1. Global homophily
    edge_homo = compute_edge_homophily(data)
    node_homo = compute_node_homophily(data)
    print(f"\n1. GLOBAL HOMOPHILY METRICS")
    print(f"   Edge Homophily: {edge_homo:.4f}")
    print(f"   Node Homophily: {node_homo:.4f}")
    print(f"   Interpretation: {edge_homo*100:.1f}% of edges connect same-class nodes")
    
    # 2. Per-class homophily
    print(f"\n2. PER-CLASS HOMOPHILY (Purity)")
    class_homo = compute_class_homophily(data)
    for c, h in sorted(class_homo.items()):
        support = (data.y == c).sum().item()
        print(f"   Class {c}: {h:.4f} (support: {support})")
    
    # 3. Label propagation ceiling
    print(f"\n3. LABEL PROPAGATION CEILING")
    print("   (Simulates structure-only classification)")
    for iters in [1, 2, 3]:
        lp_acc = compute_label_propagation_accuracy(data, iters)
        print(f"   {iters}-hop propagation accuracy: {lp_acc:.4f}")
    
    # 4. Compare to random baseline
    print(f"\n4. RANDOM BASELINE")
    num_classes = data.y.max().item() + 1
    random_acc = 1.0 / num_classes
    print(f"   Random guess accuracy: {random_acc:.4f}")
    print(f"   Structure provides {edge_homo / random_acc:.1f}x improvement over random")
    
    # 5. Theoretical explanation
    print(f"\n" + "=" * 60)
    print("INTERPRETATION")
    print("=" * 60)
    print(f"""
    The Amazon Computers dataset has edge homophily of {edge_homo:.2%}.
    
    This means: if you predict a node's class based purely on its 
    neighbors' majority class, you'd be correct ~{edge_homo:.0%} of the time.
    
    A 2-layer GCN aggregates information from 2-hop neighborhoods.
    With {edge_homo:.0%} homophily, after 2 hops of message passing,
    the "correct" class signal dominates the aggregated representation.
    
    The actual node features (BoW) become REDUNDANT because:
    1. The graph structure already encodes class information
    2. Co-purchased products are likely in the same category
    3. The GCN learns to weight neighbor signals appropriately
    
    For LR/SVM with random features: they see only noise, no structure.
    For GCN with random features: the structure does all the work.
    """)