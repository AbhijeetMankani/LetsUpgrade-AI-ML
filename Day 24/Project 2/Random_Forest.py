import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

dataset = pd.read_csv('general_data.csv')

lab_enc = LabelEncoder()

dataset.Attrition = lab_enc.fit_transform(dataset.Attrition)
dataset.BusinessTravel = lab_enc.fit_transform(dataset.BusinessTravel)
dataset.Department = lab_enc.fit_transform(dataset.Department)
dataset.EducationField = lab_enc.fit_transform(dataset.EducationField)
dataset.Gender = lab_enc.fit_transform(dataset.Gender)
dataset.JobRole = lab_enc.fit_transform(dataset.JobRole)
dataset.MaritalStatus = lab_enc.fit_transform(dataset.MaritalStatus)

to_drop = ['EmployeeCount', 'EmployeeID', 'Over18', 'StandardHours']

dataset.drop(to_drop, axis=1, inplace=True)

dataset.dropna(inplace=True)

features = dataset.drop('Attrition', axis=1)

rf_model = RandomForestClassifier(n_estimators=1000, max_features=2, oob_score=True, max_depth=5)

rf_model.fit(X = features, y = dataset.Attrition)

print("OOB SCORE:", rf_model.oob_score_)

for feature, imp in zip(features.columns, rf_model.feature_importances_):
    print(feature, imp)