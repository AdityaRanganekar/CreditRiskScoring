import os
from datetime import datetime

#common Constants
TARGET_COLUMN = "target"
PIPELINE_NAME: str = "CreditRisk"
ARTIFACT_DIR: str = "artifacts"
FILE_NAME: str = "data.csv"

TRAIN_FILE_NAME: str = "train.csv"
TEST_FILE_NAME: str = "test.csv"
SCHEMA_FILE_PATH = os.path.join("schema", "schema.yaml")

# Data Ingestion Constants
DATA_INGESTION_COLLECTION_NAME: str = "LendingClub_Records"
DATA_INGESTION_DATABASE_NAME: str = "CREDIT_DATA"
DATA_INGESTION_DIR_NAME: str = "data_ingestion"
DATA_INGESTION_FEATURE_STORE_DIR: str = "feature_store"
DATA_INGESTION_INGESTED_DIR: str = "ingested"
DATA_INGESTION_TRAIN_TEST_SPLIT_RATIO: float = 0.2