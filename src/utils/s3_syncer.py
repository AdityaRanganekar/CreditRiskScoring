import sys
import subprocess
from src.exception.exception import CreditRiskException

class S3Sync:
    def sync_folder_to_s3(self, folder: str, aws_bucket_url: str):
        try:
            command = f"aws s3 sync {folder} {aws_bucket_url}"
            subprocess.run(command, shell=True, check=True)
        except Exception as e:
            raise CreditRiskException(e, sys)

    def sync_folder_from_s3(self, folder: str, aws_bucket_url: str):
        try:
            command = f"aws s3 sync {aws_bucket_url} {folder}"
            subprocess.run(command, shell=True, check=True)
        except Exception as e:
            raise CreditRiskException(e, sys)