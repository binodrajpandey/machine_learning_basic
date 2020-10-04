import numpy as np
from tensorflow import keras

'''
X = 1, 0, 1, 2, 3, 4
Y = -3, -1, 1, 3, 5, 7
Y = 2X -1
'''
x = np.array([1, 0, 1, 2, 3, 4])
y = np.array([-3, -1, 1, 3, 5, 7])

model = keras.Sequential([keras.layers.Dense(units=1,input_shape=[1])])
model.compile(optimizer="sgd", loss='mean_squared_error')
model.fit(x,y, epochs=500)
result = model.predict([10])
print(result)