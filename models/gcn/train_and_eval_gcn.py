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
    # UPDATE: We now use the full 'train_mask' (the 10%, 50%, etc.) for training
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
    # --- Configuration ---
    # UPDATED: Using the best parameters found in your tuning
    TUNED_PARAMS = {
        'hidden_channels': 64,    
        'dropout_rate': 0.5,      
        'learning_rate': 0.01,    
        'weight_decay': 0          # Tuned value (works well with NormalizeFeatures)
    }
    
    LABEL_RATES = [0.1, 0.5, 1.0]
    SEEDS = [0, 42, 123]
    EPOCHS = 200
    EARLY_STOPPING_PATIENCE = 20   # Increased slightly for better convergence
    
    # Load Data
    data_with_holdout = torch.load('data/data_with_split.pt', weights_only=False)
    
    # CRITICAL: Keep Normalization with weight_decay=0
    transform = T.NormalizeFeatures() 
    data_with_holdout = transform(data_with_holdout)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    results = {}
    
    for rate in LABEL_RATES:
        rate_key = f"{int(rate*100)}% labels"
        print(f"\n{'='*60}\nRunning experiment with Label Rate: {rate_key}\n{'='*60}")
        
        f1_scores_for_rate = []
        reports_for_rate = []

        conf_matrices = []
        for seed in SEEDS:
            print(f"--- Seed: {seed} ---")
            
            # 1. Create the specific Training Mask (e.g., 10%)
            data_for_run = create_proportional_train_mask(data_with_holdout.clone(), rate, seed)
            
            # 2. STABLE VALIDATION STRATEGY
            # Instead of splitting the tiny training set, we use all nodes 
            # that are NOT in Train and NOT in Test as the Validation set.
            # This guarantees a large, stable validation set (~5000 nodes) for Early Stopping.
            
            # Get all indices
            all_indices = torch.arange(data_for_run.num_nodes)
            
            # Identify validation nodes (Everything that is not Train and not Test)
            # Note: 'test_mask' is the fixed holdout set from data_with_split.pt
            combined_mask = data_for_run.train_mask | data_for_run.test_mask
            
            # Create val_mask
            data_for_run.val_mask = ~combined_mask
            
            data_for_run = data_for_run.to(device)

            # --- Class Weights ---
            # Calculate weights based on the actual training nodes available
            train_labels = data_for_run.y[data_for_run.train_mask]
            class_counts = torch.bincount(train_labels)
            class_weights = 1. / class_counts.float()
            class_weights = class_weights.to(device)

            # Initialize Model
            model = GCN(data_for_run.num_features, 
                        TUNED_PARAMS['hidden_channels'], 
                        10, 
                        TUNED_PARAMS['dropout_rate']).to(device)
            
            optimizer = torch.optim.Adam(model.parameters(), 
                                         lr=TUNED_PARAMS['learning_rate'], 
                                         weight_decay=TUNED_PARAMS['weight_decay'])
            
            loss_fn = torch.nn.NLLLoss(weight=class_weights)

            # Training Loop with Early Stopping
            best_val_f1 = 0
            patience = 0
            
            for epoch in range(1, EPOCHS + 1):
                train(model, data_for_run, optimizer, loss_fn)
                
                # Evaluate on the large, stable validation set
                val_f1, _, _ = evaluate(model, data_for_run, data_for_run.val_mask)
                
                if val_f1 > best_val_f1:
                    best_val_f1 = val_f1
                    torch.save(model.state_dict(), 'best_gcn_model.pth')
                    patience = 0
                else:
                    patience += 1
                
                if patience >= EARLY_STOPPING_PATIENCE: 
                    break
            
            # Load best model and test
            model.load_state_dict(torch.load('best_gcn_model.pth', weights_only=True))
            test_f1, preds, y_test = evaluate(model, data_for_run, data_for_run.test_mask)
            f1_scores_for_rate.append(test_f1)
            print(f"  > Test Macro-F1: {test_f1:.4f}")

            # Save report for aggregation
            report = classification_report(
                y_test, preds, zero_division=0, output_dict=True
            )
            reports_for_rate.append(report)

            """
            if seed == SEEDS[0]:
                print(f"\n--- Classification Report {seed} ---")
                print(classification_report(y_test.cpu(), preds.cpu(), zero_division=0))
            """
            
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
        
        results[rate_key] = {
            'mean_f1': np.mean(f1_scores_for_rate),
            'std_f1': np.std(f1_scores_for_rate),
            "aggregated_class_metrics": aggregated_class_metrics,
            "accuracy": np.mean(accuracy_list),
            "conf_matrices": conf_matrices
        }

    print(f"\n\n{'='*60}\nFinal Aggregated Results for Tuned Vanilla GCN\n{'='*60}")
    for rate_key, result in results.items():
        print(f"\n--- {rate_key} ---")
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
        plt.title(f"GCN: Averaged Confusion Matrix — {rate_key}")
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