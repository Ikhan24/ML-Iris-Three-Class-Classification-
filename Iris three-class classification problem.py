#!/usr/bin/env python
# coding: utf-8

# In[1]:


#set up datasets
import sys
print("Python version:", sys.version)

import pandas as pd
print("pandas version:", pd.__version__)

import matplotlib
print("matplotlib version:", matplotlib.__version__)

import numpy as np
print("NumPy version:", np.__version__)

import scipy as sp
print("SciPy version:", sp.__version__)

import IPython
print("IPython version:", IPython.__version__)

import sklearn
print("scikit-learn version:", sklearn.__version__)


# In[4]:


#run first to set up the dataset
from sklearn.datasets import load_iris
iris_dataset = load_iris()


# In[6]:


print("Keys of iris_dataset:\n{}".format(iris_dataset.keys()))


# In[14]:


print(iris_dataset['DESCR'][:193] + "\n...")


# In[8]:


print("Target names:", iris_dataset['target_names'])


# In[9]:


print("Feature names:\n", iris_dataset['feature_names'])


# In[10]:


print("Type of data:", type(iris_dataset['data']))


# In[11]:


print("Shape of data:", iris_dataset['data'].shape)


# In[12]:


print("First five rows of data:\n", iris_dataset['data'][:5])


# In[15]:


print("Type of target:", type(iris_dataset['target']))


# In[16]:


print("Shape of target:", iris_dataset['target'].shape)


# In[17]:


print("Target:\n", iris_dataset['target'])


# In[16]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
    iris_dataset['data'], iris_dataset['target'], random_state=0)


# In[7]:


print("X_train shape:", X_train.shape)
print("y_train shape:", y_train.shape)


# In[8]:


print("X_test shape:", X_test.shape)
print("y_test shape:", y_test.shape)


# In[7]:


import mglearn
# create dataframe from data in X_train
# label the columns using the strings in iris_dataset.feature_names
iris_dataframe = pd.DataFrame(X_train, columns=iris_dataset.feature_names)
# create a scatter matrix from the dataframe, color by y_train
pd.plotting.scatter_matrix(iris_dataframe, c=y_train, figsize=(15, 15),
                           marker='o', hist_kwds={'bins': 20}, s=60,
                           alpha=.8, cmap=mglearn.cm3)


# In[9]:


from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=1)


# In[10]:


knn.fit(X_train, y_train)


# In[46]:


X_new = np.array([[5, 2.9, 1, 0.2]])
print("X_new.shape:", X_new.shape)


# In[49]:


prediction = knn.predict(X_new)
print("Prediction:", prediction)
print("Predicted target name:",
       iris_dataset['target_names'][prediction])


# In[13]:


y_pred = knn.predict(X_test)
print("Test set predictions:\n", y_pred)


# In[20]:


print("Test set score: {:.2f}".format(np.mean(y_pred == y_test)))


# In[21]:


print("Test set score: {:.2f}".format(knn.score(X_test, y_test)))


# In[50]:


X_train, X_test, y_train, y_test = train_test_split(
    iris_dataset['data'], iris_dataset['target'], random_state=0)

knn = KNeighborsClassifier(n_neighbors=1)
knn.fit(X_train, y_train)

print("Test set score: {:.2f}".format(knn.score(X_test, y_test)))

