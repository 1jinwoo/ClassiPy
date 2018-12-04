
# coding: utf-8

# In[48]:


from keras.models import Sequential
from keras import layers
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import SGDClassifier
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.metrics import classification_report
# file import
import data_cleaner as dc
import model_helper as mh

df = dc.clean_item_data(0)
df = dc.cleanup_categoryid(df)


# In[49]:


X = df.item_title
Y = df.categoryId


# In[50]:


X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.5)


# In[51]:



# vectorize training input data

sgd = Pipeline([('vect', CountVectorizer()),
                ('tfidf', TfidfTransformer()),
                ('clf', SGDClassifier(loss='hinge', penalty='l2',alpha=1e-3, random_state=42, max_iter=5, tol=None)),
               ])
sgd.fit(X_train, Y_train)

Y_pred = sgd.predict(X_test)

print('accuracy %s' % accuracy_score(Y_pred, Y_test))
print(classification_report(Y_test, Y_pred))

#Accuracy 81%

