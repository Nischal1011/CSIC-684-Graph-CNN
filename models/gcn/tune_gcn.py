import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
import torch_geometric.transforms as T # <--- ADDED THIS
import numpy as np
import itertools
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split

class GCN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, dropout_rate):
        super().__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        # LayerNorm helps stabilize training on Amazon datasets significantly
        self.ln = torch.nn.LayerNorm(hidden_channels) 
        self.conv2 = GCNConv(hidden_channels, out_channels)
        self.dropout_rate = dropout_rate

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = self.ln(x) # <--- Apply LayerNorm
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout_rate, training=self.training)
        x = self.conv2(x, edge_index)
        return F.log_softmax(x, dim=1)

def train(model, data, optimizer, loss_fn):
    model.train()
    optimizer.zero_grad()
    out = model(data.x, data.edge_index)
    loss = loss_fn(out[data.sub_train_mask], data.y[data.sub_train_mask])
    loss.backward()
    optimizer.step()

@torch.no_grad()
def test(model, data):
    model.eval()
    out = model(data.x, data.edge_index)
    pred = out[data.val_mask].argmax(dim=1)
    target = data.y[data.val_mask]
    val_f1 = f1_score(target.cpu(), pred.cpu(), average='macro', zero_division=0)
    return val_f1

if __name__ == "__main__":
    print("="*60)
    print("--- Tuning Vanilla GCN (Enhanced) ---")
    
    # --- Expanded Hyperparameter Grid ---
    param_grid = {
        'hidden_channels': [64],         # 64 is usually sufficient for Amazon
        'dropout_rate': [0.3, 0.5],      # Expanded
        'learning_rate': [0.01],
        # CRITICAL CHANGE: 5e-4 is often too high for Amazon. 
        # 1e-4 or 0 often works better for these rich features.
        'weight_decay': [5e-4, 1e-4, 0]  
    }
    
    TUNING_SEED = 0
    EPOCHS = 300 # Increased slightly
    EARLY_STOPPING_PATIENCE = 20 # Increased to allow recovery from bumps
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # --- CRITICAL FIX: FEATURE NORMALIZATION ---
    # This loads the data AND normalizes the rows to sum to 1.
    # This prevents the "Megaphone Effect" where high-degree nodes explode.
    data = torch.load('data/data_with_split.pt', weights_only=False)
    data = T.NormalizeFeatures()(data) 
    # -------------------------------------------
    
    # Setup Tuning Splits
    train_val_indices = torch.where(data.train_val_mask)[0].numpy()
    train_val_labels = data.y[train_val_indices].numpy()
    
    sub_train_indices, val_indices = train_test_split(
        train_val_indices, test_size=0.25, random_state=TUNING_SEED, stratify=train_val_labels
    )
    data.sub_train_mask = torch.zeros(data.num_nodes, dtype=torch.bool)
    data.sub_train_mask[sub_train_indices] = True
    data.val_mask = torch.zeros(data.num_nodes, dtype=torch.bool)
    data.val_mask[val_indices] = True
    data = data.to(device)

    # Class Weights
    train_labels = data.y[data.sub_train_mask]
    class_counts = torch.bincount(train_labels)
    class_weights = 1. / class_counts.float()
    class_weights = class_weights.to(device)

    best_val_f1 = 0
    best_params = {}
    keys, values = zip(*param_grid.items())
    combinations = [dict(zip(keys, v)) for v in itertools.product(*values)]
    
    print(f"Starting grid search over {len(combinations)} combinations...\n")

    for i, params in enumerate(combinations):
        # Print progress
        print(f"Combo {i+1}: {params}", end=" | ")
        
        model = GCN(data.num_features, params['hidden_channels'], 10, params['dropout_rate']).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=params['learning_rate'], weight_decay=params['weight_decay'])
        loss_fn = torch.nn.NLLLoss(weight=class_weights)
        
        run_best_val_f1 = 0
        patience = 0
        
        for epoch in range(1, EPOCHS + 1):
            train(model, data, optimizer, loss_fn)
            val_f1 = test(model, data)
            
            if val_f1 > run_best_val_f1:
                run_best_val_f1 = val_f1
                patience = 0
            else:
                patience += 1
            
            if patience >= EARLY_STOPPING_PATIENCE: 
                break
        
        print(f"Val F1: {run_best_val_f1:.4f}")
        
        if run_best_val_f1 > best_val_f1:
            best_val_f1 = run_best_val_f1
            best_params = params

    print("\n" + "="*60)
    print(f"Best Validation Macro F1: {best_val_f1:.4f}")
    print(f"Best Parameters: {best_params}")
    print("="*60)