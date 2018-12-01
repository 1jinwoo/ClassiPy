# libraries import
from keras.models import Sequential
from keras import layers
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from pandas import DataFrame


# file import
import item_title_cleaner as itc

df = itc.clean_item_data()
id_num = df['categoryId'].nunique()

def data_split(df, train=0.65, valid=0.15, test=0.20):
    """
    split data into training, validation, and test sets
    :param df: the data set
    :param train: percentage of training data
    :param valid: percentage of validation data
    :param test: percentage of test data
    :return: X_train, X_valid, X_test, Y_train, Y_valid, Y_test
    """

    # instantiate variables
    column_headers = list(df.columns.values)
    X_train = DataFrame(columns=column_headers)
    X_valid = DataFrame(columns=column_headers)
    X_test = DataFrame(columns=column_headers)
    Y_train = DataFrame(columns=column_headers)
    Y_valid = DataFrame(columns=column_headers)
    Y_test = DataFrame(columns=column_headers)

    id_num = df['categoryId'].nunique()
    for i in range(1, id_num+1):
        x_category_df = df.loc[df['categoryId'] == i]['item_title']
        y_category_df = df.loc[df['categoryId'] == i]['categoryId']
        x_category_train_valid, y_category_train_valid, x_category_test, y_category_test = \
            train_test_split(x_category_df,y_category_df, test_size=test)
        x_category_train, x_category_valid, y_category_train, y_category_valid = \
            train_test_split(x_category_train_valid, y_category_train_valid, train_size=train/(train+valid))
        X_train = X_train.append(x_category_train)
        X_valid = X_valid.append(x_category_valid)
        X_test = X_test.append(x_category_test)
        Y_train = Y_train.append(y_category_train)
        Y_valid = Y_valid.append(y_category_valid)
        Y_test = Y_test.append(y_category_test)

    return X_train, X_valid, X_test, Y_train, Y_valid, Y_test


# vectorize training input data
_X_train, _, _X_test, Y_train, _, Y_test = data_split(df, 0.8, 0, 0.2)
vectorizer = CountVectorizer()
X_train = vectorizer.fit_transform(_X_train)
X_test = vectorizer.transform(_X_test)

#
input_dim = X_train.shape[1] # Number of features
model = Sequential()
model.add(layers.Dense(100, input_dim=input_dim, activation='relu'))
model.add(layers.Dense(63, activation='softmax'))
model.compile(loss='mean_absolute_error',
              optimizer='adam',
              metrics=['accuracy'])
print(model.summary())
history = model.fit(X_train, Y_train,
                    epochs=1,
                    verbose=False,
                    validation_data=(X_test, Y_test),
                    batch_size=11000)

loss, accuracy = model.evaluate(X_train, Y_train, verbose=False)
print("Training Accuracy: {:.4f}".format(accuracy))
loss, accuracy = model.evaluate(X_test, Y_test, verbose=False)
print("Testing Accuracy:  {:.4f}".format(accuracy))



if __name__ == "__main__":
    print(data_split(df))




