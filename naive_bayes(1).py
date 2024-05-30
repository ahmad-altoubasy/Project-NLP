from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn import datasets

iris = datasets.load_iris()
print(iris.feature_names)
print(iris.data[0:6])
print(iris.target_names)
X = iris.data
y = iris.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4,
random_state=1)
gnb = GaussianNB()
gnb.fit(X_train, y_train)
y_pred = gnb.predict([[4.9, 3., 1.4, 0.2]])
print(y_pred)