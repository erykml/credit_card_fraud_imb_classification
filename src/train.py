import os
import pandas as pd
import mlflow

from config import PROCESSED_DIR, AUGMENTED_DIR
from scrt import *

from sklearn.ensemble import RandomForestClassifier 
from sklearn.metrics import recall_score, precision_score, f1_score

from dagshub import dagshub_logger

mlflow.set_tracking_uri("https://dagshub.com/eryk.lewinson/credit_card_fraud_imb_classification.mlflow")
os.environ['MLFLOW_TRACKING_USERNAME'] = USER_NAME
os.environ['MLFLOW_TRACKING_PASSWORD'] = PASSWORD

def get_data(name_short="raw"):
    if name_short == "raw":
        X_train = pd.read_csv(f"{PROCESSED_DIR}/X_train.csv", index_col=None)
        y_train = pd.read_csv(f"{PROCESSED_DIR}/y_train.csv", index_col=None)
    else:
        X_train = pd.read_csv(f"{AUGMENTED_DIR}/X_train_{name_short}.csv", index_col=None)
        y_train = pd.read_csv(f"{AUGMENTED_DIR}/y_train_{name_short}.csv", index_col=None)

    X_test = pd.read_csv(f"{PROCESSED_DIR}/X_test.csv", index_col=None)
    y_test = pd.read_csv(f"{PROCESSED_DIR}/y_test.csv", index_col=None)

    return X_train, X_test, y_train, y_test

if __name__ == "__main__":

    mlflow.sklearn.autolog()
    
    DATASET_NAME = "raw"
    N_ESTIMATORS = 100

    with mlflow.start_run():
        # get data
        X_train, X_test, y_train, y_test = get_data(DATASET_NAME)
        
        # fit-predict
        model = RandomForestClassifier(random_state=42,
                                       n_estimators=N_ESTIMATORS)
        model.fit(X_train, y_train.values.ravel())
        y_pred = model.predict(X_test)
        
        # calculate the scores
        recall = recall_score(y_test, y_pred)       
        precision = precision_score(y_test, y_pred)       
        f1 = f1_score(y_test, y_pred)       
       
        # logging the scores with dagshub logger
        with dagshub_logger() as logger:
            logger.log_hyperparams({
                "dataset_variant": DATASET_NAME,
                "n_estimators": N_ESTIMATORS,
            })
            logger.log_metrics(recall=recall, precision=precision, f1_score=f1)

        # logging the scores with mlflow
        mlflow.log_params({
            "dataset_variant": DATASET_NAME,
            "n_estimators": N_ESTIMATORS,
        })

        mlflow.log_metrics({
            "test_set_recall": recall,
            "test_set_precision": precision,
            "test_set_f1_score": f1,
        })