import os 
import pandas as pd 
from config import RAW_DIR, PROCESSED_DIR, AUGMENTED_DIR

from imblearn.over_sampling import RandomOverSampler, SMOTE, ADASYN 
from imblearn.under_sampling import RandomUnderSampler 

RANDOM_STATE =123

# define the considered augmentations
RESAMPLE_CONFIG = [
    {
        "name": "undersampling",
        "sampler": RandomUnderSampler(random_state=RANDOM_STATE)
    },
    {
        "name": "oversampling",
        "sampler": RandomOverSampler(random_state=RANDOM_STATE)
    },
    {
        "name": "smote",
        "sampler": SMOTE(random_state=RANDOM_STATE)
    },
    {
        "name": "adasyn",
        "sampler": ADASYN(random_state=RANDOM_STATE)
    },
]

# prepare output dir
os.makedirs(AUGMENTED_DIR, exist_ok=True)

# loading data
X_train = pd.read_csv(f"{PROCESSED_DIR}/X_train.csv", index_col=None)
y_train = pd.read_csv(f"{PROCESSED_DIR}/y_train.csv", index_col=None)

# run all augmentations
for resample_spec in RESAMPLE_CONFIG:
    
    augmenter = resample_spec["sampler"]
    X_res, y_res = augmenter.fit_resample(X_train, y_train)

    print(f"{resample_spec['name']} results ----")
    print(f"Shape before: {X_train.shape}")
    print(f"Shape after: {X_res.shape}")
    print(f"Class distribution before: {y_train.value_counts(normalize=True).values}")
    print(f"Class distribution after: {y_res.value_counts(normalize=True).values}")

    # saving the resampled data
    X_res.to_csv(f"{AUGMENTED_DIR}/X_train_{resample_spec['name']}.csv", index=None)
    y_res.to_csv(f"{AUGMENTED_DIR}/y_train_{resample_spec['name']}.csv", index=None)