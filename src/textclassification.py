import tensorflow as tf
from tensorflow import keras
import numpy as np

data = keras.datasets.imdb
(train_data, train_labels), (test_data, test_labels) = data.load_data(num_words=10000)
print(len(train_data))
word_index = data.get_word_index()
word_index = {k: (v + 3) for k, v in word_index.items()}
word_index["<PAD>"] = 0
word_index["<START>"] = 1
word_index["<UNK>"] = 2
word_index["<UNUSED>"] = 3

rev_word_index = dict([(value, key) for key, value in word_index.items()])
print(train_data.shape, test_data.shape)
print(len(train_data[1]))
train_data = keras.preprocessing.sequence.pad_sequences(train_data, value=word_index["<PAD>"], padding="post",
                                                        maxlen=250)
test_data = keras.preprocessing.sequence.pad_sequences(test_data, value=word_index["<PAD>"], padding="post", maxlen=250)
print(train_data.shape, test_data.shape)
print(len(train_data[1]))


def decode_review(index_list):
    return " ".join([rev_word_index.get(i, "?") for i in index_list])


print(test_data[0])
print(decode_review(test_data[0]))

model = keras.Sequential()
model.add(keras.layers.Embedding(10000,
                                 16))  # Creates 10000 word vectors with 16 dimension, each one represents one word. Groups the similar words
model.add(keras.layers.GlobalAveragePooling1D())
model.add(keras.layers.Dense(16, activation="relu"))
model.add(keras.layers.Dense(1, activation="sigmoid"))
model.summary()

model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
# validation data
x_val = train_data[:10000]
x_train = train_data[10000:]
y_val = train_labels[:10000]
y_train = train_labels[10000:]
model.fit(x_train, y_train, epochs=40, batch_size=512, validation_data=(x_val, y_val), verbose=1)
loss, accuracy = model.evaluate(test_data, test_labels)
print(accuracy)

test_review = test_data[0]
predict = model.predict([test_review])
print("Review:")
print(decode_review(test_review))
print("Predict: " + str(predict[0]))
print("Actual: " + str(test_labels[0]))
