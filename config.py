"""
Project configuration
"""
import os
from pathlib import Path

# Paths
PROJECT_ROOT = Path(__file__).parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
DB_PATH = f"sqlite:///{PROJECT_ROOT / 'data' / 'transactions.db'}"

# Data parameters
MIN_TRANSACTIONS_PER_PRODUCT = 20
MIN_PRICE_VARIANCE = 0.05  # 5% price variation
PRICE_CHANGE_THRESHOLD = 50  # % max price change (outlier filter)
PRICE_OUTLIER_QUANTILE = 0.99  # Per-group outlier cap quantile
MIN_PER_PRODUCT_FOR_OUTLIER = 10  # Min rows per product to use per-product bounds
HIGH_VOLUME_THRESHOLD = 50  # periods with qty >= this are "high volume"
MAX_HIGH_VOLUME_CHANGE_PCT = 300  # cap extreme changes for high-volume periods

# Elasticity model parameters
TEST_SIZE = 0.2
RANDOM_STATE = 42
ELASTICITY_THRESHOLD = -0.5  # Products more elastic than this
TRAIN_TEST_SPLIT_RATIO = 0.8  # 80/20 train/test split
CONTROL_VARIABLES = ["promotion_flag", "is_winter", "is_holiday_season"]
MIN_TRANSACTIONS_FOR_SEGMENT = 30  # Min rows per segment for aggregation

# Streamlit parameters
STREAMLIT_THEME = "light"
DEFAULT_PRICE_INCREASE_RANGE = (-20, 50)  # -20% to +50% price increase
