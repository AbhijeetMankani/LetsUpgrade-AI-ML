import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn import tree

dataset = pd.read_csv('train.csv')

dataset.Sex = LabelEncoder().fit_transform(dataset.Sex)

Features = dataset[['Age', 'Sex', 'Fare']]

tree_model = tree.DecisionTreeClassifier(max_depth=8)

tree_model.fit(X = Features, y = dataset.Survived)

print(tree_model.score(X = Features, y = dataset.Survived))

with open('Project1Tree.dot', 'w') as f:
    f = tree.export_graphviz(tree_model, feature_names=['Age', 'Sex', 'Fare'], out_file=f)