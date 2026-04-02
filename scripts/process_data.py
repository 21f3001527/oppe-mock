import pandas as pd
import numpy as np
import os
import logging
from datetime import datetime, timezone

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def process_data():
    logger.info("=== Starting Data Processing ===")

    raw_path = "data/raw/iris.csv"
    logger.info(f"Reading raw data from: {raw_path}")
    df = pd.read_csv(raw_path)

    logger.info(f"Raw shape: {df.shape}")
    logger.info(f"Columns: {df.columns.tolist()}")

    # Clean data
    initial_len = len(df)
    df = df.drop_duplicates()
    df = df.dropna()
    logger.info(f"Dropped {initial_len - len(df)} rows. Remaining: {len(df)}")

    # Feature engineering
    logger.info("Performing feature engineering...")
    df['sepal_ratio'] = df['sepal_length'] / df['sepal_width']
    df['petal_ratio'] = df['petal_length'] / df['petal_width']
    df['sepal_area']  = df['sepal_length'] * df['sepal_width']
    df['petal_area']  = df['petal_length'] * df['petal_width']

    # Required Feast columns
    df['event_timestamp'] = pd.Timestamp.now(tz='UTC')
    df['entity_id'] = range(len(df))

    # Save
    os.makedirs("processed_data", exist_ok=True)
    output_path = "processed_data/stock_data.parquet"
    df.to_parquet(output_path, index=False)

    logger.info(f"Saved to: {output_path}")
    logger.info(f"Final shape: {df.shape}")
    logger.info(f"Final columns: {df.columns.tolist()}")
    logger.info("=== Data Processing Complete ===")

if __name__ == "__main__":
    process_data()
