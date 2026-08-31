import sys
import time
from src.entity.config_entity import TrainingPipelineConfig, DataIngestionConfig
from src.components.data_ingestion import DataIngestion
from src.logging.logger import logging
from src.exception.exception import CreditRiskException

if __name__ == '__main__':
    try:
        pipeline_start_time = time.time()
        logging.info("========== TRAINING PIPELINE STARTED ==========")
        training_pipeline_config = TrainingPipelineConfig()

        # 1. Data Ingestion
        data_ingestion_config = DataIngestionConfig(training_pipeline_config)
        data_ingestion = DataIngestion(data_ingestion_config)

        logging.info("Initiate Data Ingestion")
        ingestion_start = time.time()
        data_ingestion_artifact = data_ingestion.initiate_data_ingestion()
        ingestion_end = time.time()
        
        logging.info(f"Data Ingestion completed in {ingestion_end - ingestion_start:.2f} seconds")
        print(f"Data Ingestion Artifact: {data_ingestion_artifact}")

        pipeline_end_time = time.time()
        total_time = pipeline_end_time - pipeline_start_time
        logging.info("========== TRAINING PIPELINE SUCCESS ==========")
        logging.info(f"Total Pipeline Execution Time: {total_time / 60:.2f} minutes")

    except Exception as e:
        logging.error("Pipeline execution failed.")
        raise CreditRiskException(e, sys)