
# coding: utf-8

# In[26]:


import data_cleaner as dc       
import re


# In[42]:


def clean_item_data():
    #read in file using data_cleaner
    df = dc.read_search_strings()
    
    '''
    removing . and all non-alphanumeric characters at the end of each word (e.g. 'oz.') 
    and preventing the removing of '7.5mm', except ", ', and space. 
    Æ stays, might remove later
    
    Still need to remove . after oz
    '''
    
    for index, row in df.iterrows():
        new_string = ''
        for item in row['item_title']:
            item = ''.join(c for c in item if c.isalnum() or c == '\"' or c == '\'' or c == ' ' or c == '.' or c == '$')
            
            new_string += item
        df.at[index, 'item_title']= new_string
    return df


# In[43]:


'''
Returns item_title at specified index (from 0 to 11121)
'''
def item_title(index):
    return df.loc[index]['item_title']


# In[44]:


df = clean_item_data()
for n in range(11121):
    print(n, item_title(n))

