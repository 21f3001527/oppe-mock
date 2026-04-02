#!/bin/bash
set -e

echo "========================================"
echo "  MLOps Pipeline  |  Version: v0"
echo "========================================"

# Activate virtual environment
source venv/bin/activate

# Set GCP project
export GOOGLE_CLOUD_PROJECT="meta-territory-488805-q1"
export GCLOUD_PROJECT="meta-territory-488805-q1"

echo "=== [1/7] Pulling DVC data from GCS ==="
dvc pull

echo "=== [2/7] Starting MLflow Server ==="
pkill -f "mlflow server" 2>/dev/null || true
sleep 2
mlflow server \
    --host 127.0.0.1 \
    --port 5000 \
    --backend-store-uri sqlite:///mlflow.db \
    --default-artifact-root gs://feastt/mlflow &
MLFLOW_PID=$!
echo "MLflow PID: $MLFLOW_PID"
sleep 8

echo "=== [3/7] Running DVC Pipeline ==="
dvc repro --force

echo "=== [4/7] Applying Feast Definitions ==="
cd feature_repo && feast apply && cd ..

echo "=== [5/7] Materializing Feast Features ==="
python feature_repo/materialize.py

echo "=== [6/7] Running Training ==="
python scripts/train.py

echo "=== [7/7] Pushing to GCS + GitHub ==="
dvc push

git add .
git commit -m "Pipeline run completed - data version v0" || echo "Nothing to commit"
git tag -a "run-v0" -m "Completed pipeline run for data version v0" -f
git push origin main
git push origin main --tags

echo "========================================"
echo "  Pipeline Complete! ✅"
echo "  MLflow UI: http://127.0.0.1:5000"
echo "========================================"
