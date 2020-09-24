import pandas as pd
import numpy as np
import sklearn.model_selection
import matplotlib.pyplot as plt
import pickle
from matplotlib import style

data = pd.read_csv('../student-mat.csv', sep=';')

print(data)
data = data[["G1", "G2", "G3", "studytime", "failures", "absences"]]
print(data)

predict = "G3"
X = np.array(data.drop([predict], 1))
print(X.shape)
Y = np.array(data[predict])
print(Y.shape)

'''
best = 0
for _ in range(10000):
    x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(X, Y, test_size=0.1)
    linear = linear_model.LinearRegression()
    linear.fit(x_train, y_train)
    accuracy = linear.score(x_test, y_test)
    if accuracy > best:
        best = accuracy
        with open("studentmodel.pickle", "wb") as f:
            pickle.dump(linear, f)
print("Accuracy: ", best)
'''
x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(X, Y, test_size=0.1)
pickle_in = open("studentmodel.pickle", "rb")
linear = pickle.load(pickle_in)
pickle_in.close()
coefficient = linear.coef_
print(coefficient)
intercept = linear.intercept_
print("Intercept = ", intercept)

predictions = linear.predict(x_test)

for x in range(len(predictions)):
    print(predictions[x], x_test[x], y_test[x])


# Plotting
style.use("ggplot")
plt.scatter(data.G1, data.G3) # Similary we can relate with G2, failures and studytime and absences too.
plt.show()