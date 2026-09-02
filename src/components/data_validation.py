import os
import sys
import pandas as pd
from scipy.stats import ks_2samp

from src.exception.exception import CreditRiskException
from src.logging.logger import logging
from src.entity.artifact_entity import DataIngestionArtifact, DataValidationArtifact
from src.entity.config_entity import DataValidationConfig
from src.constants import SCHEMA_FILE_PATH
from src.utils.main_utils import read_yaml_file, write_yaml_file

class DataValidation:
    def __init__(self, data_ingestion_artifact: DataIngestionArtifact,
                 data_validation_config: DataValidationConfig):
        try:
            self.data_ingestion_artifact = data_ingestion_artifact
            self.data_validation_config = data_validation_config
            self._schema_config = read_yaml_file(SCHEMA_FILE_PATH)
        except Exception as e:
            raise CreditRiskException(e, sys)

    @staticmethod
    def read_data(file_path) -> pd.DataFrame:
        try:
            return pd.read_csv(file_path)
        except Exception as e:
            raise CreditRiskException(e, sys)

    def validate_number_of_columns(self, dataframe: pd.DataFrame) -> bool:
        try:
            required_columns = [list(col.keys())[0] for col in self._schema_config["columns"]]

            missing_columns = [col for col in required_columns if col not in dataframe.columns]

            if len(missing_columns) > 0:
                logging.warning(f"Dataframe is missing these required schema columns: {missing_columns}")
                return False
            else:
                logging.info("All required schema columns are present in the dataframe.")
                return True
                
        except Exception as e:
            raise CreditRiskException(e, sys)

    def validate_numerical_columns_exist(self, dataframe: pd.DataFrame) -> bool:
        try:
            numerical_columns = self._schema_config["numerical_columns"]
            dataframe_columns = dataframe.columns
            missing_numerical_columns = [col for col in numerical_columns if col not in dataframe_columns]
                    
            if len(missing_numerical_columns) > 0:
                logging.info(f"Missing numerical columns: {missing_numerical_columns}")
                return False
                
            return True
        except Exception as e:
            raise CreditRiskException(e, sys)

    def detect_data_drift(self, base_df: pd.DataFrame, current_df: pd.DataFrame, threshold: float = 0.05) -> bool:
        try:
            status = True
            report = {}
            numerical_columns = self._schema_config["numerical_columns"]

            for column in numerical_columns:
                if column not in base_df.columns:
                    continue
                    
                d1 = base_df[column].dropna()
                d2 = current_df[column].dropna()
                
                is_same_dist = ks_2samp(d1, d2)
                # Cast the numpy boolean to a native Python boolean
                drift_found = bool(is_same_dist.pvalue < threshold)
                
                if drift_found:
                    status = False
                    
                report.update({column: {
                    "p_value": float(is_same_dist.pvalue),
                    "drift_status": drift_found
                }})
                
            drift_report_file_path = self.data_validation_config.drift_report_file_path
            write_yaml_file(file_path=drift_report_file_path, content=report)
            
            return status
        except Exception as e:
            raise CreditRiskException(e, sys)

    def initiate_data_validation(self) -> DataValidationArtifact:
        try:
            train_file_path = self.data_ingestion_artifact.trained_file_path
            test_file_path = self.data_ingestion_artifact.test_file_path

            train_dataframe = DataValidation.read_data(train_file_path)
            test_dataframe = DataValidation.read_data(test_file_path)

            train_col_status = self.validate_number_of_columns(train_dataframe)
            if not train_col_status:
                logging.warning("Train dataframe does not contain all schema columns.")

            test_col_status = self.validate_number_of_columns(test_dataframe)
            if not test_col_status:
                logging.warning("Test dataframe does not contain all schema columns.")

            num_status_train = self.validate_numerical_columns_exist(dataframe=train_dataframe)
            num_status_test = self.validate_numerical_columns_exist(dataframe=test_dataframe)

            overall_status = train_col_status and test_col_status and num_status_train and num_status_test

            drift_status = self.detect_data_drift(base_df=train_dataframe, current_df=test_dataframe)

            os.makedirs(self.data_validation_config.valid_data_dir, exist_ok=True)
            train_dataframe.to_csv(self.data_validation_config.valid_train_file_path, index=False, header=True)
            test_dataframe.to_csv(self.data_validation_config.valid_test_file_path, index=False, header=True)

            return DataValidationArtifact(
                validation_status=overall_status,
                valid_train_file_path=self.data_validation_config.valid_train_file_path,
                valid_test_file_path=self.data_validation_config.valid_test_file_path,
                invalid_train_file_path=None,
                invalid_test_file_path=None,
                drift_report_file_path=self.data_validation_config.drift_report_file_path,
            )
        except Exception as e:
            raise CreditRiskException(e, sys)