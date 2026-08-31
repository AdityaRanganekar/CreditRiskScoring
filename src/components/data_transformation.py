import os
import sys
import numpy as np
import pandas as pd
import pickle
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.exception.exception import CreditRiskException
from src.logging.logger import logging
from src.entity.artifact_entity import DataValidationArtifact, DataTransformationArtifact
from src.entity.config_entity import DataTransformationConfig
from src.utils.main_utils import read_yaml_file
from src.constants import SCHEMA_FILE_PATH
from src.utils.woe_encoder import WoEEncoder

class DataTransformation:
    def __init__(self, data_validation_artifact: DataValidationArtifact, 
                 data_transformation_config: DataTransformationConfig):
        try:
            self.data_validation_artifact = data_validation_artifact
            self.data_transformation_config = data_transformation_config
            self._schema_config = read_yaml_file(SCHEMA_FILE_PATH)
        except Exception as e:
            raise CreditRiskException(e, sys)

    def get_data_transformer_object(self) -> ColumnTransformer:
        try:
            numerical_columns = self._schema_config["numerical_columns"]
            categorical_columns = self._schema_config["categorical_columns"]

            # Numerical Pipeline: Fills missing values with the median, then standardizes the scale
            numerical_pipeline = Pipeline(steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler())
            ])

            # Categorical Pipeline: Fills missing text with 'missing', then applies custom WoE
            categorical_pipeline = Pipeline(steps=[
                ("imputer", SimpleImputer(strategy="constant", fill_value="missing")),
                ("woe_encoder", WoEEncoder(smoothing=0.5))
            ])

            # Combine pipelines into a single preprocessor
            preprocessor = ColumnTransformer(transformers=[
                ("num_pipeline", numerical_pipeline, numerical_columns),
                ("cat_pipeline", categorical_pipeline, categorical_columns)
            ])

            return preprocessor
        except Exception as e:
            raise CreditRiskException(e, sys)

    def initiate_data_transformation(self) -> DataTransformationArtifact:
        try:
            logging.info("Starting Data Transformation")
            
            # Load validated datasets
            train_df = pd.read_csv(self.data_validation_artifact.valid_train_file_path)
            test_df = pd.read_csv(self.data_validation_artifact.valid_test_file_path)

            # Isolate target column
            target_column_name = self._schema_config["target_column"][0]
            
            input_feature_train_df = train_df.drop(columns=[target_column_name], axis=1)
            target_feature_train_df = train_df[target_column_name]
            
            input_feature_test_df = test_df.drop(columns=[target_column_name], axis=1)
            target_feature_test_df = test_df[target_column_name]

            # Initialize and fit the custom preprocessor
            preprocessor = self.get_data_transformer_object()
            
            # Fit/Transform on train, only Transform on test to prevent data leakage
            input_feature_train_arr = preprocessor.fit_transform(input_feature_train_df, target_feature_train_df)
            input_feature_test_arr = preprocessor.transform(input_feature_test_df)

            # Concatenate features and target into final arrays
            train_arr = np.c_[input_feature_train_arr, np.array(target_feature_train_df)]
            test_arr = np.c_[input_feature_test_arr, np.array(target_feature_test_df)]

            # Save numpy arrays directly to disk
            os.makedirs(self.data_transformation_config.data_transformation_dir, exist_ok=True)
            os.makedirs(os.path.dirname(self.data_transformation_config.transformed_train_file_path), exist_ok=True)
            
            np.save(self.data_transformation_config.transformed_train_file_path, train_arr)
            np.save(self.data_transformation_config.transformed_test_file_path, test_arr)

            # Save the fitted preprocessor object for future inference
            os.makedirs(os.path.dirname(self.data_transformation_config.transformed_object_file_path), exist_ok=True)
            with open(self.data_transformation_config.transformed_object_file_path, "wb") as file_obj:
                pickle.dump(preprocessor, file_obj)

            logging.info("Data Transformation completed successfully.")

            return DataTransformationArtifact(
                transformed_object_file_path=self.data_transformation_config.transformed_object_file_path,
                transformed_train_file_path=self.data_transformation_config.transformed_train_file_path,
                transformed_test_file_path=self.data_transformation_config.transformed_test_file_path
            )
        except Exception as e:
            raise CreditRiskException(e, sys)