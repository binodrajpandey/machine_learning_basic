import pandas as pd
import numpy as np
import sklearn.model_selection
from sklearn import linear_model

data = pd.read_csv('../student-mat.csv', sep=';')

print(data)
data = data[["G1", "G2","G3","studytime","failures", "absences"]]
print(data)

predict = "G3"
X = np.array(data.drop([predict],1))
print(X.shape)
Y = np.array(data[predict])
print(Y.shape)
x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(X,Y,test_size=0.1)
linear = linear_model.LinearRegression()
linear.fit(x_train,y_train)
accuracy = linear.score(x_test,y_test)
print(accuracy)
coefficient = linear.coef_
print(coefficient)
intercept = linear.intercept_
print("Intercept = ",intercept)

predictions = linear.predict(x_test)

for x in range(len(predictions)):
    print(predictions[x], x_test[x], y_test[x])