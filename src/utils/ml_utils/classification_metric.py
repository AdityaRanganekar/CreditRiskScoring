import sys
from sklearn.metrics import f1_score, precision_score, recall_score
from src.exception.exception import CreditRiskException
from src.entity.artifact_entity import ClassificationMetricArtifact

def get_classification_score(y_true, y_pred) -> ClassificationMetricArtifact:
    try:
        model_f1_score = f1_score(y_true, y_pred)
        model_recall_score = recall_score(y_true, y_pred)
        model_precision_score = precision_score(y_true, y_pred)

        return ClassificationMetricArtifact(
            f1_score=model_f1_score,
            precision_score=model_precision_score, 
            recall_score=model_recall_score
        )
    except Exception as e:
        raise CreditRiskException(e, sys)