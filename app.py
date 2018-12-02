# -*- coding: utf-8 -*-
"""
Created on Sun Dec  2 00:32:24 2018

@author: Justin Won
"""

from flask import Flask
from flask import request

app = Flask(__name__)

submit_form = '<h1>ClassiPy-lookup</h1>\n<p>\n<form action="\classipy" method="post"> \nlookup: <input type="text" name="search_string">\n<input type="submit" value="Submit">\n</form>\n<p>\n'


@app.route('/', methods=['GET'])
def main():
    return submit_form

@app.route('/classipy', methods=['POST'])
def classipy():
    search_string = request.form['search_string']
        # Return model
    if search_string:
        return search_string + '<br><a href="/">GO BACK</a>'
    else:
        return submit_form

if __name__ == '__main__':
    app.run()