''' From Random Forest, we know that Income, CCAvg, Education and CD Account have a decent importance '''

import pandas as pd

from sklearn import tree

dataset = pd.read_excel('Bank_Personal_Loan_Modelling.xlsx', sheet_name=1)

dataset.dropna(inplace=True)

features = dataset[['Income', 'CCAvg', 'Education', 'CD Account']]

tree_model = tree.DecisionTreeClassifier(max_depth=6)

tree_model.fit(X = features, y = dataset['Personal Loan'])

print("Score: ", tree_model.score(X = features, y = dataset['Personal Loan']))

with open('Project3Tree.dot', 'w') as f:
    f = tree.export_graphviz(tree_model, feature_names=['Income', 'CCAvg', 'Education', 'CD Account'], out_file=f)