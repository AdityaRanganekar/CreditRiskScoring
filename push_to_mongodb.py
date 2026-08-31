import os
import sys
import pandas as pd
import numpy as np
import pymongo
from dotenv import load_dotenv

load_dotenv()
MONGO_DB_URL = os.getenv("MONGO_DB_URL")

DATABASE_NAME = "CREDIT_DATA"
COLLECTION_NAME = "LendingClub_Records"
DATA_PATH = os.path.join("credit_data", "data.csv")

if __name__ == "__main__":
    try:
        print(f"Reading dataset from {DATA_PATH}...")
        df = pd.read_csv(DATA_PATH, low_memory=False)
        
        # 1. Sanitize column names (replace spaces/special chars with underscores)
        df.columns = df.columns.str.strip().str.replace(r'[^a-zA-Z0-9_]', '_', regex=True)
        
        # 2. Ensure no columns start with a number
        df.columns = [f"col_{col}" if col[0].isdigit() else col for col in df.columns]
        
        # 3. Convert Pandas NaN to native Python None (becomes MongoDB null)
        df = df.replace({np.nan: None})
        
        records = df.to_dict(orient="records")
        
        print("Connecting to MongoDB cluster...")
        client = pymongo.MongoClient(MONGO_DB_URL)
        database = client[DATABASE_NAME]
        collection = database[COLLECTION_NAME]
        
        print("Clearing existing collection data...")
        collection.delete_many({})
        
        print(f"Inserting {len(records)} sanitized records into {DATABASE_NAME}.{COLLECTION_NAME}...")
        collection.insert_many(records)
        
        print("Upload completed successfully.")
        
    except Exception as e:
        print(f"An error occurred: {e}")
        sys.exit(1)