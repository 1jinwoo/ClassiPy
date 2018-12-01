
# coding: utf-8

# In[30]:



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


# In[43]:


'''
Returns item_title at specified index (from 0 to 11121)
'''
def item_title(index):
    return df.loc[index]['item_title']


# In[44]:

if __name__ == '__main__':
    df = clean_item_data()
    for n in range(11121):
        print(n, item_title(n))

