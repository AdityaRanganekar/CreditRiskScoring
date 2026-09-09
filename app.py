import sys
import os
import pymongo
import pandas as pd

from fastapi.middleware.cors import CORSMiddleware
from fastapi import FastAPI, File, UploadFile, Request, BackgroundTasks
from uvicorn import run as app_run
from fastapi.responses import Response, JSONResponse
from starlette.responses import RedirectResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel

import certifi
ca = certifi.where()

from dotenv import load_dotenv
load_dotenv()

from src.exception.exception import CreditRiskException
from src.logging.logger import logging
from src.pipeline.training_pipeline import TrainingPipeline
from src.utils.main_utils import load_object
from src.constants import DATA_INGESTION_COLLECTION_NAME, DATA_INGESTION_DATABASE_NAME

class CreditFeaturesSchema(BaseModel):
    loan_amnt: float
    term: str
    int_rate: float
    installment: float
    grade: str
    sub_grade: str
    emp_length: str
    home_ownership: str
    annual_inc: float
    verification_status: str
    purpose: str
    addr_state: str
    dti: float
    open_acc: float
    pub_rec: float
    revol_bal: float
    revol_util: float
    total_acc: float
    mort_acc: float
    pub_rec_bankruptcies: float

    class Config:
        extra = "allow"

mongo_db_url = os.getenv("MONGODB_URL_KEY")
client = pymongo.MongoClient(mongo_db_url, tlsCAFile=ca)
database = client[DATA_INGESTION_DATABASE_NAME]
collection = database[DATA_INGESTION_COLLECTION_NAME]

templates = Jinja2Templates(directory="./templates")
app = FastAPI(title="Credit Risk Batch Scoring API")
origins = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def run_training_in_background():
    """Runs the heavy ML pipeline safely in the background"""
    try:
        train_pipeline = TrainingPipeline()
        train_pipeline.run_pipeline()
    except Exception as e:
        logging.error(f"Background training failed: {e}")

@app.get("/", tags=["authentication"])
async def index():
    return RedirectResponse(url="/docs")

@app.get("/train")
async def train_route(background_tasks: BackgroundTasks):
    try:
        background_tasks.add_task(run_training_in_background)
        return JSONResponse(content={"message": "Training started in background. Check terminal for logs."})
    except Exception as e:
        raise CreditRiskException(e, sys)

@app.post("/predict")
async def predict_route(request: Request, file: UploadFile = File(...)):
    try:
        df = pd.read_csv(file.file)

        credit_model = load_object("final_model/model.pkl")

        y_pred = credit_model.predict(df)
        df['predicted_default'] = y_pred
        
        os.makedirs('prediction_output', exist_ok=True)
        df.to_csv('prediction_output/output.csv', index=False)

        table_html = df.head(50).to_html(classes='table table-striped', index=False)
        
        return templates.TemplateResponse(request=request, name="table.html", context={"table": table_html})
        
    except Exception as e:
        raise CreditRiskException(e, sys)

@app.post("/predict_json")
async def predict_json_route(features: CreditFeaturesSchema):
    try:
        df = pd.DataFrame([features.dict()])
        
        credit_model = load_object("final_model/model.pkl")
        y_pred = credit_model.predict(df)
        
        response = {"predicted_default": int(y_pred[0])}
        
        if hasattr(credit_model, "predict_proba"):
            y_prob = credit_model.predict_proba(df)
            response["default_probability"] = float(y_prob[0][1])
            
        return JSONResponse(content=response)
        
    except Exception as e:
        raise CreditRiskException(e, sys)
    
if __name__=="__main__":
    app_run(app, host="0.0.0.0", port=8000)