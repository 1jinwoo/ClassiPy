# -*- coding: utf-8 -*-
"""
Created on Sun Dec  2 04:05:48 2018

@author: Justin Won
"""

import Model as md

model = md.Model()

model.train_model()
print(model.get_accuracy())
print(model.get_category('lol'))