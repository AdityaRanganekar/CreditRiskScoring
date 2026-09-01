import sys
from src.pipeline.training_pipeline import TrainingPipeline
from src.exception.exception import CreditRiskException
from src.logging.logger import logging

if __name__ == "__main__":
    try:
        logging.info("Starting the Training Pipeline execution.")
        training_pipeline = TrainingPipeline()
        training_pipeline.run_pipeline()
        logging.info("Training Pipeline execution completed successfully.")
    except Exception as e:
        logging.error(f"Pipeline execution failed: {e}")
        raise CreditRiskException(e, sys)