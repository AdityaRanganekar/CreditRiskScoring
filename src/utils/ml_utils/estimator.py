import sys
from src.exception.exception import CreditRiskException

class CreditRiskModel:
    def __init__(self, preprocessor, model):
        try:
            self.preprocessor = preprocessor
            self.model = model
        except Exception as e:
            raise CreditRiskException(e, sys)
    
    def predict(self, x):
        try:
            x_transform = self.preprocessor.transform(x)
            return self.model.predict(x_transform)
        except Exception as e:
            raise CreditRiskException(e, sys)