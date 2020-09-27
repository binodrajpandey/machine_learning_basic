import sklearn
from sklearn import datasets
from sklearn import svm
from sklearn import metrics
from sklearn import model_selection

cancer = datasets.load_breast_cancer()
print(cancer.feature_names)
print(cancer.target_names)
X = cancer.data
Y = cancer.target

x_train, x_test, y_train, y_test = model_selection.train_test_split(X,Y,test_size=0.2)
classes = ["malignant", "benign"]
clf = svm.SVC(kernel="linear", C= 2)
# clf = svm.SVC(kernel="poly", degree=2)
clf.fit(x_train, y_train)
predicted = clf.predict(x_test)
acc = metrics.accuracy_score(y_test, predicted)
print(acc)