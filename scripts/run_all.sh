#!/usr/bin/env bash
set -e

###############################################################################
# ENVIRONMENT INSTRUCTIONS:
#
# BEFORE running this script, make sure you have a Python environment with
# all required packages installed.
#
# If you don’t already have one set up, do the following:
#
#   python3 -m venv .venv
#   source .venv/bin/activate
#   pip install -r requirements.txt
#
# Once your environment is active, run:
#   bash scripts/run_all.sh
#
# If you're using conda instead, run:
#   conda activate <your-env-name>
###############################################################################

echo "Running data split creation (if needed)..."
python -m utils.holdout_set_split

echo "Running GCN train+eval..."
python -m models.gcn.train_and_eval_gcn

echo "Running GCN tuning..."
python -m models.gcn.tune_gcn

echo "Running logistic train+eval..."
python -m models.logistic.train_and_eval_logistic

echo "Running logistic tuning..."
python -m models.logistic.tune_logistic

echo "Running SVM train+eval..."
python -m models.svm.train_and_eval_svm

echo "Running SVM tuning..."
python -m models.svm.tune_svm

# gcn_with_community may be uncommented-out as needed
#echo "Running community GCN eval..."
#python -m models.gcn_with_community.eval_gcn_with_community.py

#echo "Running community GCN tuning..."
#python -m models.gcn_with_community.tune_gcn_with_community.py

echo "All tasks completed! ✅"
