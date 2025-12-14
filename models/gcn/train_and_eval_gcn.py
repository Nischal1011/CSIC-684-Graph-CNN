import torch
import torch.nn.functional as F
import torch_geometric.transforms as T
from torch_geometric.nn import GCNConv
import numpy as np
from sklearn.metrics import f1_score, classification_report, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt 
from utils.data_split_utils import create_proportional_train_mask
from collections import defaultdict
from tabulate import tabulate

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
        masking_key = f"{int(100-rate*100)}% Masking"
        print(f"\n{'='*60}\nRunning experiment with Masking: {masking_key}\n{'='*60}")
        
        f1_scores_for_rate = []
        reports_for_rate = []
        conf_matrices = []
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

            # Save each of the report of each seed for aggregation 
            report = classification_report(
                y_test, preds, zero_division=0, output_dict=True
            )
            reports_for_rate.append(report)

            # Store confusion matrix for this seed
            cm = confusion_matrix(y_test, preds)
            conf_matrices.append(cm)
        
        # Aggregated class metrics 
        aggregated_class_metrics = defaultdict(lambda: defaultdict(list))
        accuracy_list = []

        # report_for_rate contains the 3 classification reports, one for each seed
        # Iterate through each reports, we'll append the value of each metrics to a list in the hashmap as correponding to their class
        #   - This list would contain all values across each seed number (i.e. "precision": [0.5, 0.43, 0.2])
        #   - Used later for ease of computing the average values across the seed number
        for rep in reports_for_rate:
            for label, metrics in rep.items():

                # Case 1: accuracy (float)
                if label == "accuracy":
                    accuracy_list.append(metrics)
                    continue

                # Case 2: per-class metrics (dict)
                if isinstance(metrics, dict):
                    for metric_name, val in metrics.items():
                        if metric_name in ["precision", "recall", "f1-score", "support"]:
                            aggregated_class_metrics[label][metric_name].append(val)
        
        results[masking_key] = {
            'mean_f1': np.mean(f1_scores_for_rate),
            'std_f1': np.std(f1_scores_for_rate),
            "aggregated_class_metrics": aggregated_class_metrics,
            "accuracy": np.mean(accuracy_list),
            "conf_matrices": conf_matrices
        }

    print(f"\n\n{'='*60}\nFinal Aggregated Results for GCN")
    if RANDOM_EMBEDDING:
        print("(Running with RANDOM EMBEDDINGS)")
    else:
        print("(Running with ORIGINAL FEATURES)")
    print(f"{'='*60}")
    for masking_key, result in results.items():
        print(f"\n--- {masking_key} ---")
        print(f"  Mean Macro-F1: {result['mean_f1']:.4f}")
        print(f"  Std Dev (stability): {result['std_f1']:.4f}")
        print(f"  Accuarcy: {result['accuracy']:.4f}")

        # Average confusion matrix for this label rate
        cms = result["conf_matrices"]
        avg_cm = np.mean(np.stack(cms), axis=0)

        # Confusion matrix 
        plt.figure(figsize=(8, 6))
        disp = ConfusionMatrixDisplay(confusion_matrix=avg_cm)
        disp.plot(cmap="Blues", values_format=".2f", colorbar=True)
        plt.title(f"GCN: Averaged Confusion Matrix — {masking_key}")
        plt.show()

        # Class metrics table
        class_metrics = result["aggregated_class_metrics"]

        table = []
        headers = ["Class", "Precision", "Recall", "F1-score", "Support"]

        for cls in sorted(class_metrics.keys(), key=str):
            prec = np.mean(class_metrics[cls]["precision"])
            rec = np.mean(class_metrics[cls]["recall"])
            f1 = np.mean(class_metrics[cls]["f1-score"])
            sup = int(np.mean(class_metrics[cls]["support"]))

            table.append([cls, f"{prec:.4f}", f"{rec:.4f}", f"{f1:.4f}", sup])

        print(tabulate(table, headers=headers, tablefmt="github"))