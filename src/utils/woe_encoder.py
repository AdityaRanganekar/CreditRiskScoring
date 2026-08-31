import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

class WoEEncoder(BaseEstimator, TransformerMixin):
    def __init__(self, smoothing=0.5):
        self.smoothing = smoothing

    def fit(self, X, y):
        X = pd.DataFrame(X)
        y = pd.Series(y)

        self.woe_dicts_ = {}
        
        total_good = (y == 1).sum() + self.smoothing
        total_bad = (y == 0).sum() + self.smoothing

        for col in X.columns:
            self.woe_dicts_[col] = {}
            unique_vals = X[col].unique()
            
            for val in unique_vals:
                mask = (X[col] == val)
                
                good_count = (y[mask] == 1).sum() + (self.smoothing / len(unique_vals))
                bad_count = (y[mask] == 0).sum() + (self.smoothing / len(unique_vals))
                
                good_dist = good_count / total_good
                bad_dist = bad_count / total_bad
                
                self.woe_dicts_[col][val] = np.log(good_dist / bad_dist)
                
        return self

    def transform(self, X):
        X_encoded = pd.DataFrame(X).copy()
        
        for col in X_encoded.columns:
            if col in self.woe_dicts_:
                X_encoded[col] = X_encoded[col].map(self.woe_dicts_[col]).fillna(0)
                
        return X_encoded.values