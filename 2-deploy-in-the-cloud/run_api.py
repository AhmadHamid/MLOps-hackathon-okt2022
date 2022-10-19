from functools import cache
from typing import List
from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import seaborn as sns
from sklearn.svm import SVC
import pickle
from google.cloud import storage

from google.cloud import bigquery
from sklearn.model_selection import train_test_split
import numpy as np

from finetune_bert import train_transformer
import os

import re

# FEATURES = "sepal_length  sepal_width  petal_length  petal_width".split()


app = FastAPI()


def save_model(clf, bucket, path):
    
    for f in ['config.json', 'pytorch_model.bin']:

        storage.Client().bucket(bucket).blob(os.path.join(path, f)).upload_from_filename(os.path.join(clf, f))
        



@cache
def load_model(bucket, path):

    from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline

    os.makedir('distilbert-aita')
    storage.Client().bucket(bucket).blob(os.path.join(path, 'pytorch_model.bin')).download_to_filename(os.path.join('distilbert_aita', 'pytorch_model.bin'))
    storage.Client().bucket(bucket).blob(os.path.join(path, 'config.json')).download_to_filename(os.path.join('distilbert_aita', 'config.json'))

    model = AutoModelForSequenceClassification.from_pretrained('distilbert_aita')
    tokenizer = AutoTokenizer.from_pretrained('distilbert-cased')
    pline = pipeline("text-classification", model=model, tokenizer=tokenizer)

    #bucket, path = re.match(r"gs://([^/]+)/(.+)", model_path).groups()
    # clf_bytes = storage.Client().bucket(bucket).blob(path).download_to_filename(destination_uri)
    # clf = pickle.loads(clf_bytes)
    return pline


class TrainRequest(BaseModel):
    model: str 


@app.post("/train")
def train_model(req: TrainRequest):


    client = bigquery.Client()
    sql = """
        SELECT text,flair
        FROM `dtumlopss.reddit_1.aita_1`
    """

    # Run a Standard SQL query using the environment's default project
    df = client.query(sql).to_dataframe()

    df.columns = ['text', 'label']
    # Run a Standard SQL query with the project set explicitly
    project_id = 'your-project-id'
    df = client.query(sql, project=project_id).to_dataframe()

    df = df.loc[df['label'].isin(['Not the A-hole', 'Asshole'])]

    train_ids, test_ids = train_test_split(np.arange(df.shape[0]))

    train_df = df.iloc[train_ids]
    test_df = df.iloc[test_ids]

    train_transformer(train_df, test_df, 'distilbert-aita')
    save_model('distilbert-aita', req.model)

    return "success"






# class TrainRequest(BaseModel):
#     dataset: str  # gs://path/to/dataset.csv
#     features: List[str]
#     target: str
#     model: str  # gs://path/to/model.pkl


# @app.post("/train")
# def train_model(req: TrainRequest):
#     dataset = pd.read_csv(req.dataset)
#     X = dataset[req.features]
#     y = dataset[req.target]
#     clf = SVC().fit(X, y)
#     save_model(clf, req.model)
#     return "success"


class PredictRequest(BaseModel):
    model: str  # gs://path/to/model.pkl
    sample: str


@app.post("/predict")
def predict(req: PredictRequest):
    pline = load_model(req.model)
    preds = pline
    return preds
