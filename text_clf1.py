import tensorflow as tf
import pandas as pd
import numpy as np
from tensorflow import keras


imdb = keras.datasets.imdb

(train_data, train_labels), (test_data, test_labels) = imdb.load_data(num_words = 10000)

word_index = imdb.get_word_index()

word_index = {k:(v+3) for k , v in word_index.items()} ## to give pad, start etc.. tags we added 3


word_index['<PAD>'] = 0
word_index['<START>'] = 1
word_index['<UNK>'] = 2
word_index['<UNUSED>'] = 3

reverse_word_index = dict([(v, k) for k, v in word_index.items()])

train_data = keras.preprocessing.sequence.pad_sequences(train_data, value = word_index['<PAD>'], padding = "post", maxlen = 250)
test_data = keras.preprocessing.sequence.pad_sequences(test_data, value = word_index['<PAD>'], padding = "post", maxlen = 250)


def decode_review(text):
    return " ".join([reverse_word_index.get(i, "?") for i in text])

model = keras.Sequential()
model.add(keras.layers.Embedding(80000, 16))
model.add(keras.layers.GlobalAveragePooling1D())
model.add(keras.layers.Dense(16, activation = "relu"))
model.add(keras.layers.Dense(1, activation = "sigmoid"))

model.summary()
model.compile(optimizer = "adam", loss= "binary_crossentropy", metrics  = ["accuracy"])

x_val = train_data[:10000]
x_train = train_data[10000:]
y_val = train_labels[:10000]
y_train = train_labels[10000:]

fitmodel = model.fit(x_train, y_train, epochs = 40, batch_size=512, verbose=1)
print(f" fitmodel ------- {fitmodel}")
results = model.evaluate(test_data, test_labels)
print(results)


print("Review:")
print(decode_review(train_data[1]))
print("Actual Classification:")
print(train_labels[1])
print("Predection")
out = model.predict_classes([train_data[1]])
print(out)

