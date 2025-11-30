import torch
import torch.nn.functional as F
import torch_geometric.transforms as T
from torch_geometric.nn import GCNConv
import numpy as np
from sklearn.metrics import f1_score, classification_report
from utils.data_split_utils import create_proportional_train_mask

RANDOM_EMBEDDING = 0

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
    loss = loss_fn(out[data.train_mask], data.y[data.train_mask])
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
    TUNED_PARAMS = {
        'hidden_channels': 64,    
        'dropout_rate': 0.5,      
        'learning_rate': 0.01,    
        'weight_decay': 0          
    }
    
    LABEL_RATES = [0.1, 0.5, 1.0]
    SEEDS = [0, 42, 123]
    EPOCHS = 200
    EARLY_STOPPING_PATIENCE = 20   
    
    # Load Data
    data_with_holdout = torch.load('data/data_with_split.pt', weights_only=False)
    
    if RANDOM_EMBEDDING:
        print("Overwriting original features with Gaussian noise.")
        
        num_nodes, num_features = data_with_holdout.x.shape
        generator = torch.Generator()
        generator.manual_seed(999) 
        data_with_holdout.x = torch.randn(num_nodes, num_features, generator=generator)
    else:
        # Only normalize if we are using the real features
        # Random noise is already effectively normalized (mean 0, std 1)
        transform = T.NormalizeFeatures() 
        data_with_holdout = transform(data_with_holdout)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    results = {}
    
    for rate in LABEL_RATES:
        rate_key = f"{int(rate*100)}% labels"
        print(f"\n{'='*60}\nRunning experiment with Label Rate: {rate_key}\n{'='*60}")
        
        f1_scores_for_rate = []
        for seed in SEEDS:
            print(f"--- Seed: {seed} ---")
            
            data_for_run = create_proportional_train_mask(data_with_holdout.clone(), rate, seed)
            
            all_indices = torch.arange(data_for_run.num_nodes)
            combined_mask = data_for_run.train_mask | data_for_run.test_mask
            data_for_run.val_mask = ~combined_mask
            
            data_for_run = data_for_run.to(device)

            train_labels = data_for_run.y[data_for_run.train_mask]
            class_counts = torch.bincount(train_labels)
            class_weights = 1. / class_counts.float()
            class_weights = class_weights.to(device)

            model = GCN(data_for_run.num_features, 
                        TUNED_PARAMS['hidden_channels'], 
                        10, 
                        TUNED_PARAMS['dropout_rate']).to(device)
            
            optimizer = torch.optim.Adam(model.parameters(), 
                                         lr=TUNED_PARAMS['learning_rate'], 
                                         weight_decay=TUNED_PARAMS['weight_decay'])
            
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
                
                if patience >= EARLY_STOPPING_PATIENCE: 
                    break
            
            model.load_state_dict(torch.load('best_gcn_model.pth', weights_only=True))
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

    print(f"\n\n{'='*60}\nFinal Aggregated Results for GCN")
    if RANDOM_EMBEDDING:
        print("(Running with RANDOM EMBEDDINGS)")
    else:
        print("(Running with ORIGINAL FEATURES)")
    print(f"{'='*60}")
    for rate_key, result in results.items():
        print(f"\n--- {rate_key} ---")
        print(f"  Mean Macro-F1: {result['mean_f1']:.4f}")
        print(f"  Std Dev (stability): {result['std_f1']:.4f}")