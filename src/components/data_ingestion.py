import os
import sys
import pymongo
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from dotenv import load_dotenv

from src.entity.config_entity import DataIngestionConfig
from src.logging.logger import logging
from src.exception.exception import CreditRiskException
from src.entity.artifact_entity import DataIngestionArtifact

load_dotenv()
MONGO_DB_URL = os.getenv("MONGO_DB_URL")

class DataIngestion:
    def __init__(self, config: DataIngestionConfig):
        try:
            self.config = config
        except Exception as e:
            raise CreditRiskException(e, sys)
        
    def export_collection_as_dataframe(self) -> pd.DataFrame:
        try:
            logging.info("Starting Data Extraction from MongoDB...")
            self.mongo_client = pymongo.MongoClient(MONGO_DB_URL)
            collection = self.mongo_client[self.config.database_name][self.config.collection_name]

            cursor = collection.find()
            df = pd.DataFrame(list(cursor))
            logging.info(f"Extracted Dataframe Shape: {df.shape}")

            if "_id" in df.columns:
                df = df.drop(columns=["_id"])
            
            df.replace({"na": np.nan}, inplace=True)
            return df
            
        except Exception as e:
            raise CreditRiskException(e, sys)

    def export_data_into_feature_store(self, dataframe: pd.DataFrame) -> pd.DataFrame:
        try:
            feature_store_file_path = self.config.feature_store_file_path
            os.makedirs(os.path.dirname(feature_store_file_path), exist_ok=True)
            dataframe.to_csv(feature_store_file_path, index=False, header=True)
            return dataframe
        except Exception as e:
            raise CreditRiskException(e, sys)

    def split_data_as_train_test(self, dataframe: pd.DataFrame):
        try:
            train_set, test_set = train_test_split(
                dataframe, 
                test_size=self.config.train_test_split_ratio, 
                stratify=dataframe["target"]
            )
            
            os.makedirs(os.path.dirname(self.config.training_file_path), exist_ok=True)
            train_set.to_csv(self.config.training_file_path, index=False, header=True)
            test_set.to_csv(self.config.testing_file_path, index=False, header=True)
            logging.info("Exported train and test files to ingested directory.")
        except Exception as e:
            raise CreditRiskException(e, sys)
    
    def initiate_data_ingestion(self) -> DataIngestionArtifact:
        try:
            dataframe = self.export_collection_as_dataframe()
            dataframe = self.export_data_into_feature_store(dataframe)
            self.split_data_as_train_test(dataframe)

            return DataIngestionArtifact(
                trained_file_path=self.config.training_file_path,
                test_file_path=self.config.testing_file_path
            )
        except Exception as e:
            raise CreditRiskException(e, sys)