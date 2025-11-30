import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def analyze_fragility(data, baseline_f1s, gcn_f1s):
    """
    data: PyG data object
    baseline_f1s: list/array of F1 scores per class for Logistic Regression
    gcn_f1s: list/array of F1 scores per class for GCN
    """
    y = data.y
    edge_index = data.edge_index
    num_classes = int(y.max()) + 1
    
    stats = []
    
    for c in range(num_classes):
        # 1. Calculate Support (Size)
        nodes_in_class = (y == c).nonzero(as_tuple=True)[0]
        support = len(nodes_in_class)
        
        # 2. Calculate Purity (Homophily)
        if support == 0:
            purity = 0
        else:
            row, col = edge_index
            # Get edges starting from nodes in this class
            mask = torch.isin(row, nodes_in_class)
            neighbors = col[mask]
            neighbor_labels = y[neighbors]
            
            if len(neighbor_labels) > 0:
                purity = (neighbor_labels == c).sum().item() / len(neighbor_labels)
            else:
                purity = 0.0
        
        # 3. Calculate Delta F1 (GCN - Baseline)
        delta_f1 = gcn_f1s[c] - baseline_f1s[c]
        
        stats.append({
            'Class': c,
            'Support': support,
            'Purity': purity,
            'Baseline_F1': baseline_f1s[c],
            'GCN_F1': gcn_f1s[c],
            'Delta_F1': delta_f1
        })
    
    df = pd.DataFrame(stats)
    
    # Sort by Delta F1 to see Winners vs Losers
    df_sorted = df.sort_values(by='Delta_F1', ascending=False)
    
    print("\n--- Class Fragility Analysis (Sorted by GCN Gain) ---")
    print(df_sorted.to_string(index=False))
    
    return df

# --- ACTUAL DATA EXECUTION ---
if __name__ == "__main__":
    # Load your real data
    data = torch.load('data/data_with_split.pt', weights_only=False)
    
    # ACTUAL VALUES EXTRACTED FROM YOUR SCREENSHOTS (10% Labels)
    
    # Baseline: Logistic Regression (Mean Macro-F1: 0.7745)
    # Class:       0       1       2       3       4       5       6       7       8       9
    baseline_f1 = [0.6614, 0.8295, 0.8770, 0.8489, 0.8767, 0.8295, 0.4546, 0.8486, 0.7642, 0.7543]
    
    # Model: Tuned Vanilla GCN (Mean Macro-F1: 0.8368)
    # Class:       0       1       2       3       4       5       6       7       8       9
    gcn_f1      = [0.8961, 0.8608, 0.9402, 0.8153, 0.8781, 0.9384, 0.5764, 0.8809, 0.7419, 0.8399]
    
    df = analyze_fragility(data, baseline_f1, gcn_f1)
    
    # Optional: Scatter Plot to visualize "Fragility"
    plt.figure(figsize=(10,6))
    
    # Scatter: X=Purity, Y=Delta F1, Size=Support (Bubble Size)
    sns.scatterplot(
        data=df, 
        x='Purity', 
        y='Delta_F1', 
        size='Support', 
        sizes=(100, 1000), # Adjusted bubble sizes for visibility
        hue='Class',
        palette='tab10',
        legend='full',
        alpha=0.7
    )
    
    # Add reference line at 0 (No Gain)
    plt.axhline(0, color='grey', linestyle='--', linewidth=1)
    
    # Add labels to bubbles
    for i in range(df.shape[0]):
        plt.text(
            df.Purity[i]+0.01, 
            df.Delta_F1[i], 
            f"C{int(df.Class[i])}", 
            horizontalalignment='left', 
            size='medium', 
            color='black', 
            weight='semibold'
        )

    plt.title("Fragility Hypothesis Validation:\nSmall Bubbles (Low Support) Appear at Extremes of Performance Gain/Loss")
    plt.xlabel("Neighborhood Purity (Homophily)")
    plt.ylabel("GCN Gain over Logistic Regression (Delta F1)")
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', title="Class ID")
    plt.tight_layout()
    plt.show()