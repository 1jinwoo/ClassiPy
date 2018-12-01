
'''
data_cleaner.py
'''

import numpy as np
import pandas as pd

def read_search_strings(file_path='search_strings.csv'):
    '''
    Reads from csv from file_path
    :return: pandas DataFrame of the csv
    '''
    df = pd.read_csv(file_path, header=0, sep=',', encoding='latin1')
    return df

read_search_strings()

def cleanup_categoryid(df):
    '''
    Assigns new category id starting from 1.
    ** This function modifies df **
    :return: dictionary[key] = categroyId
    '''
    i = 0
    category_dict = dict()
    for j, row in df.iterrows():
        category = row[3]
        if not category in category_dict.keys():
            i += 1
            category_dict[category] = i
            df.at[j, 'categoryId'] = i
        else:
            df.at[j, 'categoryId'] = i
    return category_dict


if __name__ == '__main__':
    print(cleanup_categoryid())
    df[100:200]