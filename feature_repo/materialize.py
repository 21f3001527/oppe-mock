from feast import FeatureStore
from datetime import datetime, timezone
import logging
import os

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def materialize():
    logger.info("Initializing Feast FeatureStore...")
    repo_path = os.path.dirname(os.path.abspath(__file__))
    store = FeatureStore(repo_path=repo_path)
    start = datetime(2020, 1, 1, tzinfo=timezone.utc)
    end   = datetime.now(timezone.utc)
    logger.info(f"Materializing {start} to {end}")
    store.materialize(start_date=start, end_date=end)
    logger.info("Materialization complete!")

if __name__ == "__main__":
    materialize()
