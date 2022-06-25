import pandas as pd 
from config import RAW_DIR, PROCESSED_DIR, REPORTS_DIR

from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.model_selection import train_test_split 
from sklearn.ensemble import RandomForestClassifier 

from imblearn.over_sampling import RandomOverSampler, SMOTE, ADASYN 
from imblearn.under_sampling import RandomUnderSampler 

from deepchecks.tabular import Dataset
from deepchecks.tabular.suites import full_suite

# define the considered augmentations
RESAMPLE_CONFIG = [
    {
        "name": "undersampling",
        "name_short": "rus", 
        "sampler": RandomUnderSampler(random_state=42)
    },
    {
        "name": "oversampling",
        "name_short": "ros", 
        "sampler": RandomOverSampler(random_state=42)
    },
    {
        "name": "smote",
        "name_short": "smote", 
        "sampler": SMOTE(random_state=42)
    },
    {
        "name": "adasyn",
        "name_short": "adasyn", 
        "sampler": ADASYN(random_state=42)
    },
]

# loading data
X_train = pd.read_csv(f"{PROCESSED_DIR}/X_train.csv", index_col=None)
X_test = pd.read_csv(f"{PROCESSED_DIR}/X_test.csv", index_col=None)
y_train = pd.read_csv(f"{PROCESSED_DIR}/y_train.csv", index_col=None)
y_test = pd.read_csv(f"{PROCESSED_DIR}/y_test.csv", index_col=None)

# creating a deepchecks dataset from the resampled data
ds_train = Dataset(X_train, label=y_train, cat_features=[])

# run all augmentations
for resample_spec in RESAMPLE_CONFIG:
    
    augmenter = resample_spec["sampler"]
    X_res, y_res = augmenter.fit_resample(X_train, y_train)

    print(f"{resample_spec['name']} results ----")
    print(f"Shape before: {X_train.shape}")
    print(f"Shape after: {X_res.shape}")
    print(f"Class distribution before: {y_train.value_counts(normalize=True).values}")
    print(f"Class distribution after: {y_res.value_counts(normalize=True).values}")

    # creating a deepchecks dataset from the resampled data
    ds_train_res = Dataset(X_res, label=y_res, cat_features=[])

    # generating deepchecks reports    
    suite = full_suite()
    suite = suite.run(train_dataset=ds_train, test_dataset=ds_train_res)
    suite.save_as_html(f"{REPORTS_DIR}/{resample_spec['name']}_report.html")

    # saving the resampled data
    X_train.to_csv(f"{PROCESSED_DIR}/X_train_{resample_spec['name_short']}.csv", index=None)
    X_test.to_csv(f"{PROCESSED_DIR}/X_test_{resample_spec['name_short']}.csv", index=None)