import sys
import yaml
from src.exception.exception import CreditRiskException

def read_yaml(file_path: str) -> dict:
    """
    Reads a YAML file and returns a dictionary.
    """
    try:
        with open(file_path, "r") as file:
            return yaml.safe_load(file)
    except Exception as e:
        raise CreditRiskException(e, sys)