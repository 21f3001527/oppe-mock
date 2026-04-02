import pandas as pd
import joblib
from sklearn.metrics import accuracy_score

def test_model_accuracy():
    df = pd.read_csv("data/iris.csv")

    X = df.drop("species", axis=1)
    y = df["species"]

    model = joblib.load("models/model.pkl")
    y_pred = model.predict(X)

    acc = accuracy_score(y, y_pred)

    assert acc >= 0.8
