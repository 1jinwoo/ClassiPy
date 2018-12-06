# libraries import
from keras.models import Sequential
from keras import layers
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import StratifiedKFold
import numpy as np
import pandas as pd

# file import
import data_cleaner as dc
import model_helper as mh

df = dc.clean_item_data(0)
df = dc.cleanup_categoryid(df)[0]

# fix random seed for reproducibility
seed = 7
np.random.seed(seed)


# define 5-fold cross validation test harness
X, Y = dc.data_split(df, 1, 0, 0)
kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
cvscores_train = []
cvscores_test = []
for train_valid, test in kfold.split(X, Y):
    X_Y = pd.concat([X[train_valid], Y[train_valid]], axis=1)
    _X_train, _, _X_valid, Y_train, _, Y_valid = dc.data_split(X_Y, 0.8125, 0, 0.1875)
    vectorizer = CountVectorizer(encoding='latin1')  # Allow different options (min_df, encoding)

    # convert pandas dataframes to list of strings
    x_train_list = []
    x_test_list = []
    x_valid_list = []
    for _, row in _X_train.iterrows():
        x_train_list.append(row[0])
    for _, row in _X_valid.iterrows():
        x_valid_list.append(row[0])

    vectorizer.fit(x_train_list)
    X_train = vectorizer.transform(x_train_list)
    X_test = vectorizer.transform(X[test])
    X_valid = vectorizer.transform(x_valid_list)
    Y_test = Y[test]

    # Neural Network
    print('X train shape: ' + str(X_train.shape[1]))
    input_dim = X_train.shape[1]  # Number of features
    output_dim = df['categoryId'].nunique()
    model = Sequential()
    model.add(layers.Dense(330, input_dim=input_dim, activation='relu', use_bias=False))
    model.add(layers.Dropout(rate=0.6))
    # model.add(layers.Dense(100, activation='relu'))
    model.add(layers.Dropout(rate=0.6))
    model.add(layers.Dense(output_dim, activation='softmax'))
    model.compile(loss='sparse_categorical_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])
    history = model.fit(X_train, Y_train,
                        epochs=5,
                        verbose=1,
                        validation_data=(X_valid, Y_valid),
                        batch_size=10)
    print(model.summary())

    loss, accuracy_train = model.evaluate(X_train, Y_train, verbose=False)
    print("Training Accuracy: {:.4f}".format(accuracy_train))
    loss, accuracy_test = model.evaluate(X_test, Y_test, verbose=False)
    print("Testing Accuracy:  {:.4f}".format(accuracy_test))
    mh.plot_history(history)
    cvscores_train.append(accuracy_train * 100)
    cvscores_test.append(accuracy_test * 100)

print('5-fold cross validation metrics on training set')
print("%.2f%% (+/- %.2f%%)" % (np.mean(cvscores_train), np.std(cvscores_train)))
print('5-fold cross validation metrics on testing set')
print("%.2f%% (+/- %.2f%%)" % (np.mean(cvscores_test), np.std(cvscores_test)))



