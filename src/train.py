import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import joblib
import os

os.makedirs("models", exist_ok=True)

df = pd.read_csv("data/iris.csv")

X = df.drop("species", axis=1)
y = df["species"]

model = RandomForestClassifier(random_state=42)
model.fit(X, y)

joblib.dump(model, "models/model.pkl")

print("Model trained!")
