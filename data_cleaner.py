
'''
data_cleaner.py
'''

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

def read_search_strings(file_path='search_strings.csv'):
    '''
    Reads from csv from file_path
    :return: pandas DataFrame of the csv
    '''
    df = pd.read_csv(file_path, header=0, sep=',', encoding='latin1')
    return df


def cleanup_categoryid(df):
    '''
    Assigns new category id starting from 1.
    ** This function modifies df **
    :return: dictionary[key] = categroyId
    '''
    i = -1
    category_dict = dict()
    for j, row in df.iterrows():
        category = row[3]
        if not category in category_dict.keys():
            i += 1
            category_dict[category] = i
            df.at[j, 'categoryId'] = i
        else:
            df.at[j, 'categoryId'] = i
    return df

def clean_item_data():
    #read in file using data_cleaner
    df = read_search_strings()
    
    '''
    removing . and all non-alphanumeric characters at the end of each word (e.g. 'oz.') 
    and preventing the removing of '7.5mm', except ", ', and space. 
    Æ stays, might remove later
    
    '''
    
    for index, row in df.iterrows():
        new_string = ''
        for item in row['item_title']:
            item = ''.join(c for c in item if c.isalnum() or c == '\"' or c == '\'' or c == ' ' or c == '.' or c == '$')
            
            new_string += item
        word_list = new_string.split()
        new_word = ''
        for w in word_list:
            if w.endswith('.'):
                new_word += w[:-1] + ' '
            else:
                new_word += w + ' '
        new_string = new_word
        df.at[index, 'item_title']= new_string
    return df



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
    X_train = pd.DataFrame()
    X_valid = pd.DataFrame()
    X_test = pd.DataFrame()
    Y_train = pd.DataFrame()
    Y_valid = pd.DataFrame()
    Y_test = pd.DataFrame()
    
    id_num = df['categoryId'].nunique()
    for i in range(1, id_num+1):
        x_category_df = df.loc[df['categoryId'] == i]['item_title']
        y_category_df = df.loc[df['categoryId'] == i]['categoryId']

        x_category_train_valid, x_category_test, y_category_train_valid, y_category_test = \
            train_test_split(x_category_df, y_category_df, test_size=test)
        if valid != 0:
            x_category_train, x_category_valid, y_category_train, y_category_valid = \
                train_test_split(x_category_train_valid, y_category_train_valid, train_size=train/(train+valid))
            X_train = pd.concat([X_train, x_category_train], axis=0)
            X_valid = pd.concat([X_valid, x_category_valid], axis=0)
            X_test = pd.concat([X_test, x_category_test], axis=0)
            Y_train = pd.concat([Y_train, y_category_train], axis=0)
            Y_valid = pd.concat([Y_valid, y_category_valid], axis=0)
            Y_test = pd.concat([Y_test, y_category_test], axis=0)
        else:
            X_train = pd.concat([X_train, x_category_train_valid], axis=0)
            X_test = pd.concat([X_test, x_category_test], axis=0)
            Y_train = pd.concat([Y_train, y_category_train_valid], axis=0)
            Y_test = pd.concat([Y_test, y_category_test], axis=0)

    return X_train, X_valid, X_test, Y_train, Y_valid, Y_test


if __name__ == '__main__':
    print(cleanup_categoryid())
    df[100:200]