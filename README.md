# Credit Card Fraud Classification

In this project, we attempt to identify fraudulent credit card transactions.
The dataset can be considered highly imbalanced, with only 0.17% of the observations belonging to the positive class.

As the baseline, we train a Random Forest classifier and evaluate its performance using recall, precision and the F1 Score.

In order to account for the class imbalance and to improve the model's performance, we use the following resampling approaches:
* random undersampling
* random oversampling
* SMOTE
* ADASYN

One of the issues caused by data resampling is the distortion of the relationships among the features, but also with the target. 
That is why we use `deepchecks` to investigate how the resampling impacts the distribution of the features in the training data.
Additionally, we scheduled a GitHub Action that runs every time the data or the data generating scripts are modified. 

For more voice-over, please refer to the following [article]().

If you would like to contribute to the project (for example, by exploring additional resampling approaches), please create a PR :)

**References**
* data: https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud
