# -*- coding: utf-8 -*-
"""
Created on Sun Dec  2 03:35:23 2018

@author: Justin Won
"""

# libraries import
from keras.models import Sequential
from keras import layers
from keras.models import Model
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer

# file import
import data_cleaner as dc
import model_helper as mh

class Model:
    def __init__(self, neuron=330, min_df = 0):
        self.neuron = neuron
        self.df = dc.clean_item_data(0)
        return_obj = dc.cleanup_categoryid(self.df)
        self.df = return_obj[0]
        self.category_dict = return_obj[1]

        # vectorize training input data
        _X_train, _X_valid, _X_test, self.Y_train, self.Y_valid, self.Y_test = dc.data_split(self.df, 0.65, 0.15, 0.20)
        if min_df != 0:
            self.vectorizer = CountVectorizer(encoding='latin1', min_df = min_df) # Allow different options (min_df, encoding)
        else:
            self.vectorizer = CountVectorizer(encoding='latin1') # Allow different options (min_df, encoding)

        # convert pandas dataframes to list of strings
        x_train_list = []
        x_test_list = []
        x_valid_list = []
        for _, row in _X_train.iterrows():
            x_train_list.append(row[0])
        for _, row in _X_test.iterrows():
            x_test_list.append(row[0])
        for _, row in _X_valid.iterrows():
            x_valid_list.append(row[0])

        self.vectorizer.fit(x_train_list)
        self.X_train = self.vectorizer.transform(x_train_list)
        self.X_test = self.vectorizer.transform(x_test_list)
        self.X_valid = self.vectorizer.transform(x_valid_list)
    
    def train_model(self):
        # Neural Network
        input_dim = self.X_train.shape[1] # Number of features
        output_dim = self.df['categoryId'].nunique()
        model = Sequential()
        model.add(layers.Dense(self.neuron, input_dim=input_dim, activation='relu', use_bias=False))
        model.add(layers.Dropout(rate=0.6))
        model.add(layers.Dropout(rate=0.6))
        model.add(layers.Dense(output_dim, activation='softmax'))
        model.compile(loss='sparse_categorical_crossentropy',
                      optimizer='adam',
                      metrics=['accuracy'])
        history = model.fit(self.X_train, self.Y_train,
                            epochs=4,
                            verbose=1,
                            validation_data=(self.X_valid, self.Y_valid),
                            batch_size=10)
        #print(model.summary())

        loss, self.train_accuracy = model.evaluate(self.X_train, self.Y_train, verbose=False)
        loss, self.test_accuracy = model.evaluate(self.X_test, self.Y_test, verbose=False)
        self.model = model
        
    def get_accuracy(self):
        return (round(self.train_accuracy, 4), round(self.test_accuracy, 4))
    
    
    def get_category(self,s):
        result = False
        s_arr = np.array([s])
        vector = self.vectorizer.transform(s_arr)
        prediction = self.model.predict_classes(vector)[0]
        
        for key in self.category_dict:
            if self.category_dict[key] == prediction:
                result = key
        if result:
            return result
        else:
            raise Exception('Fatal Error: Invalid model prediction')
    
    def stat(self):
        pass
    
    