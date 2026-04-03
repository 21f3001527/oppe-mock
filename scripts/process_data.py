import pandas as pd
import numpy as np
import os
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def process_data():
    logger.info("=== Starting Data Processing ===")
    df = pd.read_csv("data/raw/iris.csv")
    logger.info(f"Raw shape: {df.shape}")
    logger.info(f"Columns: {df.columns.tolist()}")

    # Clean
    df = df.drop_duplicates().dropna()
    logger.info(f"After cleaning: {df.shape}")

    # Feature engineering
    df['sepal_ratio'] = df['sepal_length'] / df['sepal_width']
    df['petal_ratio'] = df['petal_length'] / df['petal_width']
    df['sepal_area']  = df['sepal_length'] * df['sepal_width']
    df['petal_area']  = df['petal_length'] * df['petal_width']

    # Required by Feast
    df['event_timestamp'] = pd.Timestamp.now(tz='UTC')
    df['entity_id']       = range(len(df))

    os.makedirs("processed_data", exist_ok=True)
    df.to_parquet("processed_data/stock_data.parquet", index=False)
    logger.info(f"Saved: processed_data/stock_data.parquet")
    logger.info(f"Final columns: {df.columns.tolist()}")
    logger.info("=== Data Processing Complete ===")

if __name__ == "__main__":
    process_data()
