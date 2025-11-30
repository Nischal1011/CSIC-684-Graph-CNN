import torch
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from torch_geometric.utils import degree

def analyze_homophily_and_noise(data, class_names=None):
    edge_index = data.edge_index
    y = data.y
    num_classes = int(y.max()) + 1
    
    # 1. Calculate Class-wise Homophily
    class_stats = []
    
    # Adjacency check
    row, col = edge_index
    
    print(f"{'Class':<5} | {'Purity (Homophily)':<20} | {'Dominant Neighbor'}")
    print("-" * 55)

    edge_confusion_matrix = np.zeros((num_classes, num_classes))

    for c in range(num_classes):
        # Nodes belonging to class c
        nodes_in_class = (y == c).nonzero(as_tuple=True)[0]
        
        # Find all edges starting from these nodes
        # mask filters edges where source node is in class c
        mask = torch.isin(row, nodes_in_class)
        neighbors = col[mask]
        neighbor_labels = y[neighbors]
        
        # Total edges for this class
        total_edges = len(neighbor_labels)
        
        if total_edges == 0:
            purity = 0.0
            dominant_neighbor = "None"
        else:
            # Count how many neighbors are SAME class
            same_class_count = (neighbor_labels == c).sum().item()
            purity = same_class_count / total_edges
            
            # Find which OTHER class it connects to most (Noise source)
            # Remove self-loops for "noise" analysis
            other_neighbors = neighbor_labels[neighbor_labels != c]
            if len(other_neighbors) > 0:
                mode_res = torch.mode(other_neighbors)
                dominant_neighbor = f"Class {mode_res.values.item()}"
            else:
                dominant_neighbor = "Isolated"

        print(f"{c:<5} | {purity:.4f}               | {dominant_neighbor}")
        
        class_stats.append({
            'Class': c,
            'Purity': purity,
            'Support': len(nodes_in_class)
        })

        # Fill Structural Confusion Matrix
        # Rows = Source Class, Cols = Neighbor Class
        for label in neighbor_labels:
            edge_confusion_matrix[c][label.item()] += 1

    return pd.DataFrame(class_stats), edge_confusion_matrix

# --- RUNNING THE ANALYSIS ---
if __name__ == "__main__":
    # Load your data
    data = torch.load('data/data_with_split.pt', weights_only=False)
    
    df_stats, struct_cm = analyze_homophily_and_noise(data)
    
    # --- VISUALIZATION 1: Structural Confusion Matrix ---
    # This shows "Who is talking to whom"
    # Normalize by row to show probabilities
    row_sums = struct_cm.sum(axis=1)
    norm_struct_cm = struct_cm / row_sums[:, np.newaxis]
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(norm_struct_cm, annot=True, fmt='.2f', cmap='Reds')
    plt.title("Structural Confusion Matrix\n(Probability of Node U connecting to Class V)")
    plt.xlabel("Neighbor Class")
    plt.ylabel("Source Class")
    plt.show()
    
    # Print the DataFrame for your report
    print(df_stats)