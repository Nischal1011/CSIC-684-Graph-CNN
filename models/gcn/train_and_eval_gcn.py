import torch
import torch.nn.functional as F
from torch_geometric.datasets import Amazon
from torch_geometric.nn import GCNConv
import numpy as np
from sklearn.metrics import f1_score, classification_report
from sklearn.model_selection import train_test_split
from utils import create_proportional_train_mask

# --- GCN Model Definition ---
class GCN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, dropout_rate):
        super().__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, out_channels)
        self.dropout_rate = dropout_rate

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout_rate, training=self.training)
        x = self.conv2(x, edge_index)
        return F.log_softmax(x, dim=1)

# --- Training and Evaluation Functions ---
def train(model, data, optimizer, loss_fn):
    model.train()
    optimizer.zero_grad()
    out = model(data.x, data.edge_index)
    loss = loss_fn(out[data.sub_train_mask], data.y[data.sub_train_mask])
    loss.backward()
    optimizer.step()

@torch.no_grad()
def evaluate(model, data, mask):
    model.eval()
    out = model(data.x, data.edge_index)
    pred = out[mask].argmax(dim=1)
    target = data.y[mask]
    score = f1_score(target.cpu(), pred.cpu(), average='macro', zero_division=0)
    return score, pred, target

if __name__ == "__main__":
    # --- Configuration ---
    TUNED_PARAMS = {
        'hidden_channels': 64,    
        'dropout_rate': 0.5,      
        'learning_rate': 0.01,    
        'weight_decay': 5e-4
    }
    LABEL_RATES = [0.1, 0.5, 1.0]
    SEEDS = [0, 42, 123]
    EPOCHS = 200
    EARLY_STOPPING_PATIENCE = 10

    data_with_holdout = torch.load('data/data_with_split.pt', weights_only=False)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    results = {}
    for rate in LABEL_RATES:
        rate_key = f"{int(rate*100)}% labels"
        print(f"\n{'='*60}\nRunning experiment with Label Rate: {rate_key}\n{'='*60}")
        
        f1_scores_for_rate = []
        for seed in SEEDS:
            print(f"--- Seed: {seed} ---")
            
            data_for_run = create_proportional_train_mask(data_with_holdout.clone(), rate, seed)
            
            # Create a temporary val set from the train mask for early stopping
            train_indices = torch.where(data_for_run.train_mask)[0].numpy()
            train_labels = data_for_run.y[train_indices].numpy()
            if len(np.unique(train_labels)) < 10: # Ensure all classes are in the small split
                print("  > Skipping seed due to missing classes in a small validation split.")
                continue
            sub_train_indices, val_indices = train_test_split(
                train_indices, test_size=0.25, random_state=seed, stratify=train_labels
            )
            data_for_run.sub_train_mask = torch.zeros(data_for_run.num_nodes, dtype=torch.bool)
            data_for_run.sub_train_mask[sub_train_indices] = True
            data_for_run.val_mask = torch.zeros(data_for_run.num_nodes, dtype=torch.bool)
            data_for_run.val_mask[val_indices] = True
            data_for_run = data_for_run.to(device)

            # --- Class Weights for Imbalance ---
            sub_train_labels = data_for_run.y[data_for_run.sub_train_mask]
            class_counts = torch.bincount(sub_train_labels)
            class_weights = 1. / class_counts.float()
            class_weights = class_weights.to(device)

            model = GCN(data_for_run.num_features, TUNED_PARAMS['hidden_channels'], 10, TUNED_PARAMS['dropout_rate']).to(device)
            optimizer = torch.optim.Adam(model.parameters(), lr=TUNED_PARAMS['learning_rate'], weight_decay=TUNED_PARAMS['weight_decay'])
            loss_fn = torch.nn.NLLLoss(weight=class_weights)

            best_val_f1 = 0
            patience = 0
            for epoch in range(1, EPOCHS + 1):
                train(model, data_for_run, optimizer, loss_fn)
                val_f1, _, _ = evaluate(model, data_for_run, data_for_run.val_mask)
                if val_f1 > best_val_f1:
                    best_val_f1 = val_f1
                    torch.save(model.state_dict(), 'best_gcn_model.pth')
                    patience = 0
                else:
                    patience += 1
                if patience >= EARLY_STOPPING_PATIENCE: break
            
            model.load_state_dict(torch.load('best_gcn_model.pth'))
            test_f1, preds, y_test = evaluate(model, data_for_run, data_for_run.test_mask)
            f1_scores_for_rate.append(test_f1)
            print(f"  > Test Macro-F1: {test_f1:.4f}")

            if seed == SEEDS[0]:
                print("\n--- Classification Report (Seed 0) ---")
                print(classification_report(y_test.cpu(), preds.cpu(), zero_division=0))
        
        results[rate_key] = {
            'mean_f1': np.mean(f1_scores_for_rate),
            'std_f1': np.std(f1_scores_for_rate)
        }

    print(f"\n\n{'='*60}\nFinal Aggregated Results for Tuned Vanilla GCN\n{'='*60}")
    for rate_key, result in results.items():
        print(f"\n--- {rate_key} ---")
        print(f"  Mean Macro-F1: {result['mean_f1']:.4f}")
        print(f"  Std Dev (stability): {result['std_f1']:.4f}")