# MLOps OPPE Pipeline - Quick Reference

## Environment
- GCP_PROJECT_ID: meta-territory-488805-q1
- GCS_BUCKET: feastt
- MLflow port: 5001
- Data version: v0
- Dataset: data/raw/iris.csv
- Target: species

## To run from scratch on exam day:
cd ~/oppe-mock
source venv/bin/activate

# Terminal 1 - Start MLflow
mlflow server --host 127.0.0.1 --port 5001 --backend-store-uri sqlite:///mlflow.db --default-artifact-root gs://feastt/mlflow &

# Terminal 2 - Run pipeline
python scripts/process_data.py
cd feature_repo && feast apply && cd ..
python feature_repo/materialize.py
python scripts/train.py
dvc push
git add . && git commit -m "run" && git tag -a "run-v0" -f -m "run-v0" && git push origin main --tags
