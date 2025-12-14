# logistic.py
import torch
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score, classification_report, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
from utils.data_split_utils import create_proportional_train_mask
from collections import defaultdict
from tabulate import tabulate

RANDOM_EMBEDDING = 0  # Set to 1 to replace features with random noise (Hypothesis 2), 0 for normal features

if __name__ == "__main__":
    TUNED_PARAMS = {
        'C': 0.01, 
        'max_iter': 1000,
        'random_state': 42,
        'class_weight': 'balanced' 
    }

    LABEL_RATES = [0.1, 0.5, 1.0] 
    SEEDS = [0, 42, 123]

    # Load the data file with the 80/20 split
    data_with_holdout = torch.load('data/data_with_split.pt', weights_only=False)
    
    if RANDOM_EMBEDDING:
        print("Overwriting original features with Gaussian noise.")
        
        # Get dimensions of the original data
        num_nodes, num_features = data_with_holdout.x.shape
        
        # Generate random noise (Standard Normal Distribution)
        # We set a fixed generator seed here to ensure the 'randomness' is consistent across runs
        # if you want to reproduce the specific noise pattern.
        generator = torch.Generator()
        generator.manual_seed(999) 
        
        # Replace features
        data_with_holdout.x = torch.randn(num_nodes, num_features, generator=generator)
    # ----------------------------------------------------

    # Store all results
    results = {}

    for rate in LABEL_RATES:
        masking_key = f"{int(100-rate*100)}% Masking"
        print(f"\n{'='*60}\nRunning experiment with Masking: {masking_key}\n{'='*60}")
        
        f1_scores_for_rate = []
        reports_for_rate = []
        conf_matrices = []
        for seed in SEEDS:
            print(f"--- Seed: {seed} ---")

            # Create the specific training mask for this run
            data_for_run = create_proportional_train_mask(
                data_with_holdout.clone(), 
                label_rate=rate, 
                seed=seed
            )

            # Prepare data for scikit-learn
            X_train = data_for_run.x[data_for_run.train_mask].numpy()
            y_train = data_for_run.y[data_for_run.train_mask].numpy()
            X_test = data_for_run.x[data_for_run.test_mask].numpy()
            y_test = data_for_run.y[data_for_run.test_mask].numpy()

            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            
            # Train the model using the tuned hyperparameters
            model = LogisticRegression(**TUNED_PARAMS)
            model.fit(X_train_scaled, y_train)

            # Evaluate on the held-out test set
            preds = model.predict(X_test_scaled)
            macro_f1 = f1_score(y_test, preds, average='macro', zero_division=0)
            f1_scores_for_rate.append(macro_f1)
            
            print(f"Test Macro-F1: {macro_f1:.4f}")
            
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

    print(f"\n\n{'='*60}\nFinal Aggregated Results for Logistic Regression\n")
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
        plt.title(f"LogReg: Averaged Confusion Matrix — {masking_key}")
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