import os
import sys
import mlflow
import mlflow.sklearn
import dagshub
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import GridSearchCV

from src.exception.exception import CreditRiskException
from src.logging.logger import logging
from src.entity.artifact_entity import DataTransformationArtifact, ModelTrainerArtifact
from src.entity.config_entity import ModelTrainerConfig
from src.utils.main_utils import load_object, load_numpy_array_data, save_object
from src.utils.ml_utils.classification_metric import get_classification_score
from src.utils.ml_utils.estimator import CreditRiskModel

class ModelTrainer:
    def __init__(self, data_transformation_artifact: DataTransformationArtifact, 
                 model_trainer_config: ModelTrainerConfig):
        try:
            self.model_trainer_config = model_trainer_config
            self.data_transformation_artifact = data_transformation_artifact
        except Exception as e:
            raise CreditRiskException(e, sys)

    def track_mlflow(self, best_model, train_metric, test_metric):
        dagshub.init(repo_owner='AdityaRanganekar', repo_name='CreditRiskScoring', mlflow=True)
        with mlflow.start_run():
            mlflow.log_params(best_model.get_params())

            mlflow.log_metric("train_f1_score", train_metric.f1_score)
            mlflow.log_metric("train_precision", train_metric.precision_score)
            mlflow.log_metric("train_recall", train_metric.recall_score)

            mlflow.log_metric("test_f1_score", test_metric.f1_score)
            mlflow.log_metric("test_precision", test_metric.precision_score)
            mlflow.log_metric("test_recall", test_metric.recall_score)
            
            mlflow.sklearn.log_model(
                sk_model=best_model, 
                name = "model",
                skops_trusted_types=["xgboost.core.Booster", "xgboost.sklearn.XGBClassifier"]
            )

    def evaluate_models(self, X_train, y_train, X_test, y_test, models, params):
        try:
            report = {}
            for i in range(len(list(models))):
                model_name = list(models.keys())[i]
                model = list(models.values())[i]
                param = params[model_name]

                # GridSearchCV safely handles multiprocessing across your CPU cores
                gs = GridSearchCV(model, param, cv=3, scoring='f1')
                gs.fit(X_train, y_train)

                # Train the optimal configuration
                model.set_params(**gs.best_params_)
                model.fit(X_train, y_train)

                # Evaluate and store the F1 score
                y_test_pred = model.predict(X_test)
                test_model_score = get_classification_score(y_test, y_test_pred).f1_score
                report[model_name] = test_model_score

            return report
        except Exception as e:
            raise CreditRiskException(e, sys)

    def train_model(self, X_train, y_train, X_test, y_test):
        try:
            models = {
                "Logistic Regression": LogisticRegression(class_weight="balanced", max_iter=1000),
                "Random Forest": RandomForestClassifier(class_weight="balanced", n_jobs=1),
                "XGBoost": XGBClassifier(scale_pos_weight=3, n_jobs=1)
            }

            params = {
                    "Logistic Regression": {
                        'C': [0.1, 1.0],  
                        'solver': ['lbfgs']
                    },
                    "Random Forest": {
                        'n_estimators': [128],  
                        'max_depth': [10, 15]
                    },
                    "XGBoost": {
                        'learning_rate': [0.1, 0.05], 
                        'max_depth': [3, 5, 7],          
                        'n_estimators': [128, 200],
                        'subsample': [0.8] 
                    }
                }

            model_report = self.evaluate_models(X_train, y_train, X_test, y_test, models, params)

            best_model_score = max(sorted(model_report.values()))
            best_model_name = list(model_report.keys())[list(model_report.values()).index(best_model_score)]
            best_model = models[best_model_name]
            
            logging.info(f"Best Model Found: {best_model_name} with F1-score: {best_model_score}")

            if best_model_score < self.model_trainer_config.expected_accuracy:
                raise Exception("No best model found below expected accuracy threshold.")

            y_train_pred = best_model.predict(X_train)
            classification_train_metric = get_classification_score(y_train, y_train_pred)

            y_test_pred = best_model.predict(X_test)
            classification_test_metric = get_classification_score(y_test, y_test_pred)

            # Overfitting / Underfitting Guardrail
            metric_diff = abs(classification_train_metric.f1_score - classification_test_metric.f1_score)
            if metric_diff > self.model_trainer_config.overfitting_underfitting_threshold:
                raise Exception(
                    f"Model rejected due to overfitting/underfitting. "
                    f"Train-Test score difference ({metric_diff:.4f}) exceeds the threshold ({self.model_trainer_config.overfitting_underfitting_threshold})."
                )

            self.track_mlflow(best_model, classification_train_metric, classification_test_metric)

            preprocessor = load_object(file_path=self.data_transformation_artifact.transformed_object_file_path)
            
            os.makedirs(os.path.dirname(self.model_trainer_config.trained_model_file_path), exist_ok=True)
            credit_model_obj = CreditRiskModel(preprocessor, best_model)
            save_object(self.model_trainer_config.trained_model_file_path, obj=credit_model_obj)

            os.makedirs("final_model", exist_ok=True)
            save_object("final_model/model.pkl", obj=credit_model_obj)

            return ModelTrainerArtifact(
                trained_model_file_path=self.model_trainer_config.trained_model_file_path,
                train_metric_artifact=classification_train_metric,
                test_metric_artifact=classification_test_metric
            )
        except Exception as e:
            raise CreditRiskException(e, sys)

    def initiate_model_trainer(self) -> ModelTrainerArtifact:
        try:
            train_arr = np.load(self.data_transformation_artifact.transformed_train_file_path)
            test_arr = np.load(self.data_transformation_artifact.transformed_test_file_path)

            x_train, y_train, x_test, y_test = (
                train_arr[:, :-1],
                train_arr[:, -1],
                test_arr[:, :-1],
                test_arr[:, -1],
            )

            return self.train_model(x_train, y_train, x_test, y_test)
        except Exception as e:
            raise CreditRiskException(e, sys)