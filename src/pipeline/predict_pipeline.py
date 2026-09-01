import sys
import pandas as pd
from src.exception.exception import CreditRiskException
from src.utils.main_utils import load_object

class PredictPipeline:
    def __init__(self):
        self.model_path = "final_model/model.pkl"

    def predict(self, features: pd.DataFrame):
        try:
            model = load_object(file_path=self.model_path)

            predictions = model.predict(features)
            probabilities = model.predict_proba(features)[:, 1] if hasattr(model, "predict_proba") else None
            
            return predictions, probabilities
        
        except Exception as e:
            raise CreditRiskException(e, sys)

class CustomData:
    def __init__(self, 
                 loan_amnt: float,
                 term: str,
                 int_rate: float,
                 installment: float,
                 grade: str,
                 emp_length: str,
                 home_ownership: str,
                 annual_inc: float,
                 verification_status: str,
                 purpose: str,
                 dti: float,
                 delinq_2yrs: float,
                 inq_last_6mths: float,
                 open_acc: float,
                 pub_rec: float,
                 revol_bal: float,
                 revol_util: float,
                 total_acc: float,
                 initial_list_status: str,
                 application_type: str,
                 mort_acc: float):

        self.loan_amnt = loan_amnt
        self.term = term
        self.int_rate = int_rate
        self.installment = installment
        self.grade = grade
        self.emp_length = emp_length
        self.home_ownership = home_ownership
        self.annual_inc = annual_inc
        self.verification_status = verification_status
        self.purpose = purpose
        self.dti = dti
        self.delinq_2yrs = delinq_2yrs
        self.inq_last_6mths = inq_last_6mths
        self.open_acc = open_acc
        self.pub_rec = pub_rec
        self.revol_bal = revol_bal
        self.revol_util = revol_util
        self.total_acc = total_acc
        self.initial_list_status = initial_list_status
        self.application_type = application_type
        self.mort_acc = mort_acc

    def get_data_as_data_frame(self) -> pd.DataFrame:
        try:
            custom_data_input_dict = {
                "loan_amnt": [self.loan_amnt],
                "term": [self.term],
                "int_rate": [self.int_rate],
                "installment": [self.installment],
                "grade": [self.grade],
                "emp_length": [self.emp_length],
                "home_ownership": [self.home_ownership],
                "annual_inc": [self.annual_inc],
                "verification_status": [self.verification_status],
                "purpose": [self.purpose],
                "dti": [self.dti],
                "delinq_2yrs": [self.delinq_2yrs],
                "inq_last_6mths": [self.inq_last_6mths],
                "open_acc": [self.open_acc],
                "pub_rec": [self.pub_rec],
                "revol_bal": [self.revol_bal],
                "revol_util": [self.revol_util],
                "total_acc": [self.total_acc],
                "initial_list_status": [self.initial_list_status],
                "application_type": [self.application_type],
                "mort_acc": [self.mort_acc]
            }

            return pd.DataFrame(custom_data_input_dict)
        except Exception as e:
            raise CreditRiskException(e, sys)