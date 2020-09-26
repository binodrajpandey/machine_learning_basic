import pandas as pd
from sklearn import preprocessing, model_selection
from sklearn.neighbors import KNeighborsClassifier
data = pd.read_csv('../car.data')
print(data.head())
le = preprocessing.LabelEncoder()
buying = le.fit_transform(list(data["buying"]))
maint = le.fit_transform(list(data["maint"]))
doors = le.fit_transform(list(data["doors"]))
persons = le.fit_transform(list(data["persons"]))
lug_boot = le.fit_transform(list(data["buying"]))
safety = le.fit_transform(list(data["safety"]))
cls = le.fit_transform(list(data["class"]))
print(buying)
print(maint)

X = list(zip(buying,maint,doors,persons,lug_boot, safety))
Y = list(cls)

x_train, x_test, y_train, y_test = model_selection.train_test_split(X,Y,test_size=0.1)

model = KNeighborsClassifier(n_neighbors=9)
model.fit(x_train,y_train)
accuracy = model.score(x_test,y_test)
print(accuracy)

predicted = model.predict(x_test)
classes = ["unacc", "acc", "good", "vgood"]
for x in range(len(x_test)):
    print("Predicted: ", classes[predicted[x]],"Actual: ", classes[y_test[x]])
    n = model.kneighbors([x_test[x]], 9, True)
    print("Neighbours:", n)