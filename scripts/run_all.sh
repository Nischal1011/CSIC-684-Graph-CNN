set -e

echo "Running GCN train+eval..."
python -m models.gcn.train_and_eval_gcn

echo "Running logistic train+eval..."
python -m models.logistic.train_and_eval_logistic

echo "Running SVM train+eval..."
python -m models.svm.train_and_eval_svm


