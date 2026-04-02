from feast import FeatureStore
from datetime import datetime, timezone
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def materialize():
    logger.info("Initializing Feast FeatureStore...")
    store = FeatureStore(repo_path="feature_repo")

    start_date = datetime(2020, 1, 1, tzinfo=timezone.utc)
    end_date   = datetime.now(timezone.utc)

    logger.info(f"Materializing from {start_date} to {end_date}")
    store.materialize(start_date=start_date, end_date=end_date)
    logger.info("Materialization complete!")

if __name__ == "__main__":
    materialize()
