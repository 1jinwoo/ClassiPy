
# coding: utf-8

# In[11]:


import numpy as np
import pandas as pd

def read_search_strings():
    df = pd.read_csv('search_strings.csv', header=0, sep=',', encoding='latin1')
    return df

read_search_strings()

