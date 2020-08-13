import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split

train = pd.read_csv('train.csv')

train.drop(['Ticket', 'Cabin', 'PassengerId', 'Name', 'Age', 'Fare'], axis=1, inplace=True)

lab_enc = LabelEncoder()

train.Sex = lab_enc.fit_transform(train.Sex)
train.Embarked = lab_enc.fit_transform(train.Embarked)

def acc_score_and_confusion_matrix(label):
    clf = GaussianNB()
    
    X = train.drop(label, axis = 1)
    Y = train[label]
    
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, random_state = True, test_size = 0.2)
    
    clf.fit(X_train, Y_train)
    
    y_pred = clf.predict(X_test)
    
    print("Accuracy Score: \n", accuracy_score(Y_test, y_pred=y_pred, normalize=True))
    
    print("\nConfusion Matrix: \n", confusion_matrix(Y_test, y_pred=y_pred))
    
for i in train.columns:
    print('\n\n', i)
    acc_score_and_confusion_matrix(i)