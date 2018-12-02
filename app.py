# -*- coding: utf-8 -*-
"""
Created on Sun Dec  2 00:32:24 2018

@author: Justin Won
"""

from flask import Flask
from flask import request

import tensorflow as tf
import Model as md
import os

app = Flask(__name__)

submit_form = '<h1>ClassiPy-lookup</h1>\n<p>\n<form action="\classipy" method="post"> \nlookup: <input type="text" name="search_string">\n<input type="submit" value="Submit">\n</form>\n<p>\n'
accuracies = None
global graph,model

@app.route('/', methods=['GET'])
def main():
    return submit_form

@app.route('/classipy', methods=['POST'])
def classipy():
    search_string = request.form['search_string']
        # Return model
    with graph.as_default():
        if search_string:
            result_string = model.get_category(search_string)
            train_acc = int(accuracies[0] * 100)
            test_acc = int(accuracies[1] * 100)
            status_msg = f'Search String: {search_string} <br> Search Result: {result_string} <br> Training Accuracy: {train_acc}% <br> Testing Accuracy: {test_acc}% <br>'
            return status_msg + '<br><a href="/">GO BACK</a>'
        else:
            return submit_form

if __name__ == '__main__':
    model = md.Model()
    model.train_model()
    accuracies = model.get_accuracy()
    graph = tf.get_default_graph()
    app.run()