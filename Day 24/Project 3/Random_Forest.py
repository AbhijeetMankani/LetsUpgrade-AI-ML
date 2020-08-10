import pandas as pd

from sklearn.ensemble import RandomForestClassifier

dataset = pd.read_excel('Bank_Personal_Loan_Modelling.xlsx', sheet_name=1)

to_drop = ['ID', 'ZIP Code']

dataset.drop(to_drop, axis=1, inplace=True)

features = dataset.drop('Personal Loan', axis=1)

rf_model = RandomForestClassifier(n_estimators=1000, max_features=2, oob_score=True, max_depth=6)

rf_model.fit(X = features, y = dataset["Personal Loan"])

print("OOB SCORE:", rf_model.oob_score_)

for feature, imp in zip(features.columns, rf_model.feature_importances_):
    print(feature, imp)