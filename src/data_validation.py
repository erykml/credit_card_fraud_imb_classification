import pandas as pd 
import glob 
import os

from config import RAW_DIR, PROCESSED_DIR, REPORTS_DIR, AUGMENTED_DIR

from deepchecks.tabular import Dataset
from deepchecks.tabular.suites import full_suite

# prepare output dir for the data validation reports
os.makedirs(REPORTS_DIR, exist_ok=True)

# get the names of the resampled datasets
data_files = glob.glob(f"{AUGMENTED_DIR}/*.csv")

resampled_names = [file_name.replace(f"{AUGMENTED_DIR}/", "").replace("X_train_", "").replace(".csv", "") for file_name in data_files if "X_train_" in file_name]
resampled_names = [name for name in resampled_names if len(name) > 0]

# loading the original data
X_train = pd.read_csv(f"{PROCESSED_DIR}/X_train.csv", index_col=None)
y_train = pd.read_csv(f"{PROCESSED_DIR}/y_train.csv", index_col=None)

# creating a deepchecks dataset from the original data
ds_train = Dataset(X_train, label=y_train, cat_features=[])

for resampled_data in resampled_names:

    print(f"Validating the {resampled_data} dataset ----")

    X_res = pd.read_csv(f"{AUGMENTED_DIR}/X_train_{resampled_data}.csv", index_col=None)
    y_res = pd.read_csv(f"{AUGMENTED_DIR}/y_train_{resampled_data}.csv", index_col=None)

    print(f"X: {X_res.shape}")
    print(f"y: {y_res.shape}")

    # creating a deepchecks dataset from the resampled data
    ds_train_res = Dataset(X_res, label=y_res, cat_features=[])

    # generating deepchecks reports    
    suite = full_suite()
    suite = suite.run(train_dataset=ds_train, test_dataset=ds_train_res)
    suite.save_as_html(f"{REPORTS_DIR}/{resampled_data}_report.html")

