# libraries import
from keras.models import Sequential
from keras import layers
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
import pandas as pd

# file import
import data_cleaner as dc

df = dc.clean_item_data()
df = dc.cleanup_categoryid(df)

# vectorize training input data
_X_train, _, _X_test, Y_train, _, Y_test = dc.data_split(df, 0.7, 0.1, 0.2)
vectorizer = CountVectorizer()
print(_X_train)

x_train_list = []
x_test_list = []
for _, row in _X_train.iterrows():
    x_train_list.append(row[0])

for _, row in _X_test.iterrows():
    x_test_list.append(row[0])

vectorizer.fit(x_train_list)
X_train = vectorizer.transform(x_train_list)
X_test = vectorizer.transform(x_test_list)

# Neural Network
input_dim = X_train.shape[1] # Number of features
model = Sequential()
model.add(layers.Dense(100, input_dim=input_dim, activation='relu'))
model.add(layers.Dense(1, activation='softmax'))
model.compile(loss='mean_absolute_error',
              optimizer='adam',
              metrics=['accuracy'])
print(model.summary())
history = model.fit(X_train, Y_train,
                    epochs=10,
                    verbose=False,
                    validation_data=(X_test, Y_test),
                    batch_size=5000)

loss, accuracy = model.evaluate(X_train, Y_train, verbose=False)
print("Training Accuracy: {:.4f}".format(accuracy))
loss, accuracy = model.evaluate(X_test, Y_test, verbose=False)
print("Testing Accuracy:  {:.4f}".format(accuracy))
