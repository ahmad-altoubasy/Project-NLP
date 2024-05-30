import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn import metrics
my_data = pd.read_csv("Tree_data.csv")
X = my_data.values[:,0:4]
y = my_data.values[:,4]
clf = DecisionTreeClassifier()
clf.fit(X,y)
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size = 0.3)
y_pred = clf.predict(X_test)
print("y_pred")
print(y_pred)
print("y_test")
print(y_test)
print(metrics.accuracy_score(y_test, y_pred))
print(clf.predict([[1,2,1,0]]))