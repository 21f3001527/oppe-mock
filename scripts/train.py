import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
from feast import FeatureStore
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, classification_report
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials
import joblib, os, logging
from datetime import datetime, timezone

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

MLFLOW_URI      = os.environ.get("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000")
EXPERIMENT_NAME = "iris_classification"

def fetch_features():
    logger.info("Fetching features from Feast...")
    store = FeatureStore(repo_path="feature_repo")
    df = pd.read_parquet("processed_data/stock_data.parquet")
    entity_df = pd.DataFrame({
        "entity_id":       df["entity_id"].tolist(),
        "event_timestamp": [datetime.now(timezone.utc)] * len(df)
    })
    features = store.get_historical_features(
        entity_df=entity_df,
        features=[
            "iris_features:sepal_length",
            "iris_features:sepal_width",
            "iris_features:petal_length",
            "iris_features:petal_width",
            "iris_features:sepal_ratio",
            "iris_features:petal_ratio",
            "iris_features:sepal_area",
            "iris_features:petal_area",
            "iris_features:species",
        ]
    ).to_df()
    logger.info(f"Features fetched: {features.shape}")
    return features

def train():
    logger.info("=== Starting Training ===")
    mlflow.set_tracking_uri(MLFLOW_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)
    df = fetch_features()
    feature_cols = ["sepal_length","sepal_width","petal_length","petal_width",
                    "sepal_ratio","petal_ratio","sepal_area","petal_area"]
    X = df[feature_cols].values
    y = df["species"].values
    X_train,X_test,y_train,y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y)
    space = {
        "n_estimators":      hp.choice("n_estimators",      [50,100,150,200]),
        "max_depth":         hp.choice("max_depth",         [3,5,7,10]),
        "min_samples_split": hp.choice("min_samples_split", [2,4,6,8]),
        "min_samples_leaf":  hp.choice("min_samples_leaf",  [1,2,3,4]),
    }
    best_run = {"accuracy": 0, "params": {}, "model": None}
    def objective(params):
        with mlflow.start_run(nested=True):
            model = RandomForestClassifier(random_state=42, **params)
            model.fit(X_train, y_train)
            preds = model.predict(X_test)
            acc = float(accuracy_score(y_test, preds))
            f1  = float(f1_score(y_test, preds, average="weighted"))
            mlflow.log_params(params)
            mlflow.log_metric("accuracy", acc)
            mlflow.log_metric("f1_score", f1)
            if acc > best_run["accuracy"]:
                best_run.update({"accuracy": acc, "params": params.copy(), "model": model})
        return {"loss": -acc, "status": STATUS_OK}
    with mlflow.start_run(run_name="iris_rf_hyperopt"):
        fmin(fn=objective, space=space, algo=tpe.suggest, max_evals=10, trials=Trials())
        best = best_run["model"]
        preds = best.predict(X_test)
        acc = float(accuracy_score(y_test, preds))
        f1  = float(f1_score(y_test, preds, average="weighted"))
        mlflow.log_params(best_run["params"])
        mlflow.log_metric("best_accuracy", acc)
        mlflow.log_metric("best_f1_score", f1)
        os.makedirs("models", exist_ok=True)
        joblib.dump(best, "models/best_model.pkl")
        mlflow.sklearn.log_model(best, artifact_path="model",
                                 registered_model_name="IrisClassifier")
        report = classification_report(y_test, preds)
        with open("models/classification_report.txt","w") as f:
            f.write(report)
        mlflow.log_artifact("models/classification_report.txt")
        logger.info(f"Best Accuracy: {acc:.4f}")
    logger.info("=== Training Complete ===")

if __name__ == "__main__":
    train()
