import torch
from sklearn.model_selection import train_test_split

def create_proportional_train_mask(data, label_rate, seed):
    """
    Creates a new 'train_mask' by proportionally sampling from the 'train_val_mask'.
    """
    if not hasattr(data, 'train_val_mask'):
        raise ValueError("Data object must have a 'train_val_mask' from the hold-out split.")

    train_val_indices = torch.where(data.train_val_mask)[0].numpy()
    train_val_labels = data.y[train_val_indices].numpy()

    if label_rate == 1.0:
        train_indices = train_val_indices
    else:
        train_indices, _ = train_test_split(
            train_val_indices,
            train_size=label_rate,
            random_state=seed,
            stratify=train_val_labels
        )

    data.train_mask = torch.zeros(data.num_nodes, dtype=torch.bool)
    data.train_mask[train_indices] = True

    return data