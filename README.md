# (CISC-684-Graph-CNN) Understanding When Graph Convolutional Networks Help: A Diagnostic Study on Label Scarcity and Structural Properties

Research codebase for experiments comparing **Graph Convolutional Networks (GCNs)** with feature-based baselines (Logistic Regression and SVM) on product co-purchase graphs (Amazon Computers dataset). The analyses investigate **when and why GCNs help**, focusing on simulated label scarcity, feature ablation, and per-class analysis.

---

## 📂 Repository Overview

- **`data/` (Expected)**  
  Contains the processed PyTorch Geometric dataset.

  - `data_with_split.pt`: PyG `Data` object with precomputed train/validation/test splits.

- **`models/`**  
  Training and evaluation scripts for all models considered in this project.

  - **`gcn/`**
    - `train_and_eval_gcn.py`: Runs GCN experiments across multiple random seeds and label rates.
    - `tune_gcn.py`: Performs a grid search over GCN hyperparameters by training a normalized two-layer GCN on a stratified train/validation split and selecting the configuration that achieves the highest validation macro-F1 score with early stopping.
  - **`logistic/`**
    - `train_and_eval_logistic.py`: Logistic Regression baseline experiments.
    - `tune_logistic.py`: Tunes a Logistic Regression baseline by standardizing node features, performing cross-validated grid search over the regularization strength on a full-label training subset, and selecting the configuration that maximizes macro-F1 score.
  - **`svm/`**
    - `train_and_eval_svm.py`: SVM baseline experiments.
    - `tune_svm.py`: Tunes a linear SVM baseline by standardizing node features and performing cross-validated grid search over the regularization parameter `C` on the full training set, selecting the value that maximizes macro-F1 score.

- **`utils/`**  
  Helper functions for dataset preparation and experimental setup.

  - `holdout_set_split.py`: Creates an 80/20 stratified hold-out split and saves it to `data/data_with_split.pt`.
  - `data_split_utils.py`: Generates proportional training masks for label-rate experiments.

- **`analysis/`**  
  Scripts for hypothesis evaluation of experimental results.

- **`figures/`**  
  Scripts for generating plots and figures used in analysis and reporting.

- **`requirements.txt`**  
  Python dependencies required to run the project.

---

## 🚀 Getting started

1. Create a virtual environment and install dependencies (PowerShell):

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

2. Prepare the dataset.

- If you already have `data/data_with_split.pt`, skip this step.
- To create the hold-out split from the raw Amazon dataset, run:

```powershell
python -m utils.holdout_set_split
```

This will produce `data/data_with_split.pt` containing `train_val_mask` and `test_mask`.

3. Run experiments

- Option 1: Run all experiments sequentially (Recommended)

  The `run_all.sh` script executes the GCN, Logistic Regression, and SVM training/evaluation scripts.

  ```bash
  bash run_all.sh
  ```

- Option 2: Run individual experiments

  ```powershell
  python -m models.gcn.train_and_eval_gcn
  python -m models.logistic.train_and_eval_logistic
  python -m models.svm.train_and_eval_svm
  ```

4.  Run the scripts in `analysis` and `figures` to process the results generated in Step 3, producing visualizations and further evaluation.

---

## 📐 Design Notes

- **Label Rate Simulation:** The training scripts create per-run `train_mask` values which sample proportionally from `train_val_mask` for different label rates (10%, 50%, 100%).
- **Ablation Study:** `train_and_eval_*` scripts support a `RANDOM_EMBEDDING` toggle that replaces node features with Gaussian noise to test structure-only hypotheses.
