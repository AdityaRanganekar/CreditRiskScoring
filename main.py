import sys
import time
from src.entity.config_entity import TrainingPipelineConfig, DataIngestionConfig, DataValidationConfig, DataTransformationConfig, ModelTrainerConfig
from src.components.data_ingestion import DataIngestion
from src.components.data_validation import DataValidation
from src.components.data_transformation import DataTransformation
from src.components.model_trainer import ModelTrainer
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

        # 2. Data Validation
        logging.info("Initiating Data Validation")
        validation_start = time.time()
        
        data_validation_config = DataValidationConfig(training_pipeline_config)
        data_validation = DataValidation(data_ingestion_artifact, data_validation_config)
        data_validation_artifact = data_validation.initiate_data_validation()
        
        validation_end = time.time()
        logging.info(f"Data Validation completed in {validation_end - validation_start:.2f} seconds")
        print(f"Data Validation Artifact: {data_validation_artifact}")

        # 3. Data Transformation
        logging.info("Initiating Data Transformation")
        transformation_start = time.time()
        
        data_transformation_config = DataTransformationConfig(training_pipeline_config)
        data_transformation = DataTransformation(data_validation_artifact, data_transformation_config)
        data_transformation_artifact = data_transformation.initiate_data_transformation()
        
        transformation_end = time.time()
        logging.info(f"Data Transformation completed in {transformation_end - transformation_start:.2f} seconds")
        print(f"Data Transformation Artifact: {data_transformation_artifact}")

        # 4. Model Trainer
        logging.info("Initiating Model Trainer")
        trainer_start = time.time()
        
        model_trainer_config = ModelTrainerConfig(training_pipeline_config)
        model_trainer = ModelTrainer(data_transformation_artifact, model_trainer_config)
        model_trainer_artifact = model_trainer.initiate_model_trainer()
        
        trainer_end = time.time()
        logging.info(f"Model Trainer completed in {trainer_end - trainer_start:.2f} seconds")
        print(f"Model Trainer Artifact: {model_trainer_artifact}")

        pipeline_end_time = time.time()
        total_time = pipeline_end_time - pipeline_start_time
        logging.info("========== TRAINING PIPELINE SUCCESS ==========")
        logging.info(f"Total Pipeline Execution Time: {total_time / 60:.2f} minutes")

    except Exception as e:
        logging.error("Pipeline execution failed.")
        raise CreditRiskException(e, sys)