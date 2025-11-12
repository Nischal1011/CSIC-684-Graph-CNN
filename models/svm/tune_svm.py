import torch
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from utils import create_proportional_train_mask

if __name__ == "__main__":
    print("="*60)
    print("--- Tuning Support Vector Machine (SVM) ---")
    
    # Use a fixed seed and a large label rate for stable tuning
    TUNING_SEED = 0
    TUNING_LABEL_RATE = 1.0 # Tune on the full 80% of data
    
    data = torch.load('data/data_with_split.pt',weights_only=False)
    data_for_tuning = create_proportional_train_mask(data, TUNING_LABEL_RATE, TUNING_SEED)

    X_train = data_for_tuning.x[data_for_tuning.train_mask].numpy()
    y_train = data_for_tuning.y[data_for_tuning.train_mask].numpy()

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)

    # Define the parameter grid to search for SVM
    param_grid = {
        'C': [0.001, 0.01, 0.1, 1, 10, 100]
    }
    
    # Set up the grid search with 3-fold cross-validation
    grid_search = GridSearchCV(
        estimator=SVC(kernel='linear', random_state=42, class_weight='balanced'),
        param_grid=param_grid,
        scoring='f1_macro', # We care about Macro F1
        cv=3, # Use 3-fold cross-validation
        verbose=1 # Show progress
    )
    
    # Run the search
    grid_search.fit(X_train_scaled, y_train)

    print("\n" + "="*60)
    print("--- Tuning Complete ---")
    print(f"Best cross-validation Macro F1: {grid_search.best_score_:.4f}")
    print(f"Best Parameters: {grid_search.best_params_}")
    print("Use this 'C' value in train_and_eval_svm.py")
    print("="*60)