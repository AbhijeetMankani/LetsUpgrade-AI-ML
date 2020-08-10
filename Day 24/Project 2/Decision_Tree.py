''' From Random Forest, we know that Age, TotalWorkingYears, YearsAtCompany and YearsWithCurrManager have a decent importance '''

import pandas as pd

from sklearn import tree

from sklearn.preprocessing import LabelEncoder

dataset = pd.read_csv('general_data.csv')

lab_enc = LabelEncoder()

dataset.Attrition = lab_enc.fit_transform(dataset.Attrition)

dataset.dropna(inplace=True)

features = dataset[['Age', 'TotalWorkingYears', 'YearsAtCompany', 'YearsWithCurrManager']]

tree_model = tree.DecisionTreeClassifier(max_depth=5)

tree_model.fit(X = features, y = dataset.Attrition)

print("Score: ", tree_model.score(X = features, y = dataset.Attrition))

with open('Project2Tree.dot', 'w') as f:
    f = tree.export_graphviz(tree_model, feature_names=['Age', 'TotalWorkingYears', 'YearsAtCompany', 'YearsWithCurrManager'], out_file=f)