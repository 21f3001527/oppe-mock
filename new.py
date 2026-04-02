from sklearn.datasets import load_iris
import pandas as pd
import os

os.makedirs("data", exist_ok=True)

iris = load_iris()

df = pd.DataFrame(
    iris.data,
    columns=["sepal_length","sepal_width","petal_length","petal_width"]
)
df["species"] = iris.target

df.to_csv("data/iris.csv", index=False)