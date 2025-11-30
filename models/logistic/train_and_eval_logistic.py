# logistic.py
import torch
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score, classification_report
from utils.data_split_utils import create_proportional_train_mask

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
        rate_key = f"{int(rate*100)}% labels"
        print(f"\n{'='*60}\nRunning experiment with Label Rate: {rate_key}\n{'='*60}")
        
        f1_scores_for_rate = []
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
            
            # Print classification report for the first seed only
            if seed == SEEDS[0]:
                print("\n Classification Report (Seed 0)")
                print(classification_report(y_test, preds, zero_division=0))
        
        # Store aggregated results for this rate
        results[rate_key] = {
            'mean_f1': np.mean(f1_scores_for_rate),
            'std_f1': np.std(f1_scores_for_rate)
        }

    print(f"\n\n{'='*60}\nFinal Aggregated Results for Logistic Regression\n")
    if RANDOM_EMBEDDING:
        print("(Running with RANDOM EMBEDDINGS)")
    else:
        print("(Running with ORIGINAL FEATURES)")
    print(f"{'='*60}")
    
    for rate_key, result in results.items():
        print(f"\n--- {rate_key} ---")
        print(f"  Mean Macro-F1: {result['mean_f1']:.4f}")
        print(f"  Std Dev (stability): {result['std_f1']:.4f}")