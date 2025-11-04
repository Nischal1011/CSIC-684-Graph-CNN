# tune_feature_centric_gcn.py
import torch
import torch.nn.functional as F
from torch_geometric.utils import to_networkx
from torch_geometric.nn import GCNConv
import networkx as nx
import community as community_louvain
import numpy as np
import itertools
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split

# --- 1. MODIFIED GCN MODEL & HELPER FUNCTIONS ---

class GCN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, dropout_rate):
        super().__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, out_channels)
        self.dropout_rate = dropout_rate

    def forward(self, x, edge_index):
        # The first GCN layer produces the intermediate embeddings
        embeddings = self.conv1(x, edge_index)
        x = F.relu(embeddings)
        x = F.dropout(x, p=self.dropout_rate, training=self.training)
        
        # The second layer produces the final predictions
        predictions = self.conv2(x, edge_index)
        
        # Return both for our new, more complex loss function
        return F.log_softmax(predictions, dim=1), embeddings

@torch.no_grad()
def detect_communities(data, device):
    print("  Detecting communities using Louvain algorithm...")
    graph = to_networkx(data, to_undirected=True)
    partition = community_louvain.best_partition(graph)
    num_nodes = data.num_nodes
    community_assignments = torch.zeros(num_nodes, dtype=torch.long)
    for node_idx, community_id in partition.items():
        community_assignments[node_idx] = community_id
    return community_assignments.to(device)

@torch.no_grad()
def get_feature_centroids(data, community_assignments):
    num_communities = community_assignments.max().item() + 1
    num_features = data.num_features
    device = data.x.device
    centroids = torch.zeros(num_communities, num_features, device=device)
    centroids.scatter_add_(0, community_assignments.unsqueeze(1).expand(-1, num_features), data.x)
    community_sizes = torch.bincount(community_assignments).float().unsqueeze(1)
    centroids = centroids / community_sizes.clamp(min=1)
    return centroids.to(device)

# --- 2. NEW TRAINING AND EVALUATION FUNCTIONS ---

def train(model, data, optimizer, loss_fn, comm_assign, centroids, centroid_weight):
    model.train()
    optimizer.zero_grad()
    
    predictions, embeddings = model(data.x, data.edge_index)
    
    # Loss 1: Standard Classification Loss
    classification_loss = loss_fn(predictions[data.sub_train_mask], data.y[data.sub_train_mask])
    
    # Loss 2: Feature-Centric Regularization
    # Part A: Prediction Consistency
    centroid_embeds = F.relu(model.conv1(centroids, data.edge_index)) # This is an approximation
    centroid_preds = F.log_softmax(model.conv2(centroid_embeds), dim=1)
    community_target_preds = centroid_preds[comm_assign].detach()
    prediction_loss = F.kl_div(predictions, community_target_preds, reduction='batchmean', log_target=True)

    # Part B: Embedding Consistency
    node_target_embeddings = centroids[comm_assign].detach()
    # Pass node embeddings through an activation function for a fair comparison
    active_embeddings = F.relu(embeddings)
    embedding_loss = F.mse_loss(active_embeddings, node_target_embeddings)
    
    # Combine all three losses
    total_loss = classification_loss + (centroid_weight * (prediction_loss + embedding_loss))
    
    total_loss.backward()
    optimizer.step()

@torch.no_grad()
def test(model, data):
    model.eval()
    predictions, _ = model(data.x, data.edge_index)
    pred = predictions[data.val_mask].argmax(dim=1)
    target = data.y[data.val_mask]
    val_f1 = f1_score(target.cpu(), pred.cpu(), average='macro', zero_division=0)
    return val_f1

# --- 3. MAIN TUNING SCRIPT ---

if __name__ == "__main__":
    print("="*60)
    print("--- Tuning Feature-Centric Community GCN ---")
    
    param_grid = {
        'hidden_channels': [32, 64],
        'dropout_rate': [0.25, 0.5],
        'learning_rate': [0.01],
        'weight_decay': [5e-4],
        'centroid_weight': [0.01, 0.1, 0.5] # The crucial new hyperparameter
    }
    
    TUNING_SEED = 0
    EPOCHS = 200
    EARLY_STOPPING_PATIENCE = 10
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    data = torch.load('data/data_with_split.pt', weights_only=False)
    
    train_val_indices = torch.where(data.train_val_mask)[0].numpy()
    train_val_labels = data.y[train_val_indices].numpy()
    sub_train_indices, val_indices = train_test_split(train_val_indices, test_size=0.25, random_state=TUNING_SEED, stratify=train_val_labels)
    data.sub_train_mask = torch.zeros(data.num_nodes, dtype=torch.bool); data.sub_train_mask[sub_train_indices] = True
    data.val_mask = torch.zeros(data.num_nodes, dtype=torch.bool); data.val_mask[val_indices] = True
    
    comm_assign = detect_communities(data, device)
    centroids = get_feature_centroids(data, comm_assign)
    data = data.to(device)

    train_labels = data.y[data.sub_train_mask]
    class_counts = torch.bincount(train_labels)
    class_weights = (1. / class_counts.float()).to(device)

    best_val_f1 = 0
    best_params = {}
    keys, values = zip(*param_grid.items())
    combinations = [dict(zip(keys, v)) for v in itertools.product(*values)]
    print(f"\nStarting grid search over {len(combinations)} combinations...\n")

    for i, params in enumerate(combinations):
        print(f"--- Combo {i+1}/{len(combinations)}: {params} ---")
        model = GCN(data.num_features, params['hidden_channels'], 10, params['dropout_rate']).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=params['learning_rate'], weight_decay=params['weight_decay'])
        loss_fn = torch.nn.NLLLoss(weight=class_weights)
        
        run_best_val_f1 = 0
        patience = 0
        for epoch in range(1, EPOCHS + 1):
            train(model, data, optimizer, loss_fn, comm_assign, centroids, params['centroid_weight'])
            val_f1 = test(model, data)
            if val_f1 > run_best_val_f1:
                run_best_val_f1 = val_f1; patience = 0
            else: patience += 1
            if patience >= EARLY_STOPPING_PATIENCE: break
        
        print(f"  > Best Validation F1: {run_best_val_f1:.4f}")
        if run_best_val_f1 > best_val_f1:
            best_val_f1 = run_best_val_f1; best_params = params
            print("  *** New best parameters found! ***")

    print("\n" + "="*60)
    print("--- Tuning Complete ---")
    print(f"Best Validation Macro F1: {best_val_f1:.4f}")
    print(f"Best Parameters: {best_params}")
    print("Use these parameters in train_and_eval_feature_centric_gcn.py")
    print("="*60)