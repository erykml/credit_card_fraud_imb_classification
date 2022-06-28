import pandas as pd 
from config import RAW_DIR, PROCESSED_DIR
import os

from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import train_test_split 

# load data
df = pd.read_csv(f"{RAW_DIR}/creditcard.csv")

# separate the target
X = df.copy() 
y = X.pop("Class")

# dropping unnecessary feature
X = X.drop(columns=["Time"])

# stratified train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y,  
                                                    random_state=42,
                                                    test_size=0.2,  
                                                    stratify=y) 

# sanity check 
print(y_train.value_counts(normalize=True).values)
print(y_test.value_counts(normalize=True).values)

# scale the amount feature
robust_scaler = RobustScaler()
X_train[["Amount"]] = robust_scaler.fit_transform(X_train[["Amount"]])
X_test[["Amount"]] = robust_scaler.transform(X_test[["Amount"]])

# prepare output dir
os.makedirs(PROCESSED_DIR, exist_ok=True)

# saving data
X_train.to_csv(f"{PROCESSED_DIR}/X_train.csv", index=None)
X_test.to_csv(f"{PROCESSED_DIR}/X_test.csv", index=None)
y_train.to_csv(f"{PROCESSED_DIR}/y_train.csv", index=None)
y_test.to_csv(f"{PROCESSED_DIR}/y_test.csv", index=None)