import pandas as pd

def test_data_schema():
    df = pd.read_csv("data/iris.csv")

    expected_columns = [
        "sepal_length", "sepal_width",
        "petal_length", "petal_width", "species"
    ]

    assert list(df.columns) == expected_columns


def test_no_missing():
    df = pd.read_csv("data/iris.csv")
    assert df.isnull().sum().sum() == 0
