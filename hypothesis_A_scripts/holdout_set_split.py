# prepare_holdout_split.py

import torch
from torch_geometric.datasets import Amazon
from sklearn.model_selection import train_test_split
import numpy as np
from collections import Counter

def verify_split_distribution(data, mask, dataset_name):
    """Helper function to print the class distribution for a given mask."""
    labels = data.y[mask].numpy()
    total_samples = len(labels)
    counts = Counter(labels)
    
    print(f"\n Class Distribution for: {dataset_name} ({total_samples} nodes)")
    # Sort by class index for consistent ordering
    for class_id in sorted(counts.keys()):
        count = counts[class_id]
        percentage = (count / total_samples) * 100
        print(f"  Class {class_id}: {count:<5} nodes ({percentage:.2f}%)")

if __name__ == "__main__":
    # We will split the data into 80% for training/validation and 20% for a fixed hold-out test set.
    TEST_SIZE = 0.2
    
    # Using a fixed random state ensures that this split is the same every time the script is run.
    RANDOM_STATE = 42
    dataset = Amazon(root='./data/AmazonComputers', name='Computers')
    data = dataset[0]
    print("Raw dataset loaded.")
    print(f"  Total nodes: {data.num_nodes}")
    
    # We need a list of indices to split, from 0 to num_nodes - 1.
    indices = np.arange(data.num_nodes)
    labels = data.y.numpy()

    # The 'stratify' argument to ensure that the split preserves the same percentage of samples for each class.
    train_val_indices, test_indices = train_test_split(
        indices,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=labels
    )
    
    print(f"\nPerformed stratified split ({100 - TEST_SIZE*100:.0f}% train/val, {TEST_SIZE*100:.0f}% test).")

    # These masks will be the single source of truth for all future experiments.
    data.train_val_mask = torch.zeros(data.num_nodes, dtype=torch.bool)
    data.train_val_mask[train_val_indices] = True

    data.test_mask = torch.zeros(data.num_nodes, dtype=torch.bool)
    data.test_mask[test_indices] = True
    
    print("\nVerifying the split integrity")
    
    # Check 1: No overlap between sets
    assert not (data.train_val_mask & data.test_mask).any(), "Error: Overlap found between train/val and test sets!"
    print("  > Integrity Check 1/3: PASSED - No overlap between train/val and test sets.")
    
    # Check 2: All nodes are accounted for
    assert data.train_val_mask.sum() + data.test_mask.sum() == data.num_nodes, "Error: Node count mismatch!"
    print(f"  > Integrity Check 2/3: PASSED - All {data.num_nodes} nodes are accounted for.")
    
    # Check 3: Class distribution is preserved
    verify_split_distribution(data, torch.ones(data.num_nodes, dtype=torch.bool), "Original Dataset")
    verify_split_distribution(data, data.train_val_mask, "80% Train/Val Pool")
    verify_split_distribution(data, data.test_mask, "20% Hold-Out Test Set")
    print("\n  > Integrity Check 3/3: PASSED - Class distributions are preserved (compare percentages above).")

    output_path = 'data/data_with_split.pt'
    torch.save(data, output_path)
    
    print(f"   File saved to: {output_path}")
