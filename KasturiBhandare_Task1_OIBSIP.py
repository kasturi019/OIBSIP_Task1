#!/usr/bin/env python
# coding: utf-8

# # Importing the necessary libraries

# In[42]:


import numpy as np
import pandas as pd


# In[43]:


from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split


# In[44]:


import seaborn as sns
import matplotlib.pyplot as plt


# # Reading the dataset

# In[45]:


df = pd.read_csv(r'C:\Users\Kasturi\Desktop\Oasis\Iris.csv')
df


# In[46]:


df.shape


# In[47]:


df.info()


# In[48]:


df.describe()


# In[49]:


df.drop('Id', axis=1, inplace = True)


# In[50]:


df['Species'].value_counts()


# # Visualizing the dataset

# In[51]:


sns.countplot(df['Species']);


# In[52]:


plt.bar(df['Species'], df['PetalWidthCm'])


# In[53]:


sns.pairplot(df,hue='Species', diag_kind="hist")


# In[54]:


df.rename(columns={'SepalLengthCm': 'Sepal_length', 'SepalWidthCm': 'Sepal_width', 
                   'PetalWidthCm': 'Petal_width', 'PetalLengthCm': 'Petal_length','Species': 'Species'})


# # Splitting the dataset into training and testing

# In[55]:


x = df.drop(['Species'], axis = 1)


# In[56]:


x


# In[57]:


y = df["Species"]
y


# In[58]:


x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.4,random_state=0)


# In[59]:


x_test


# In[60]:


x_test.size


# In[61]:


x_train.size


# In[62]:


y_test.size


# In[63]:


y_train.size


# # Evaluating the training dataset

# In[64]:


dtc_model = DecisionTreeClassifier()
dtc_model.fit(x_train, y_train)


# In[65]:


y_pred_train = dtc_model.predict(x_train)


# In[66]:


Accuracy = accuracy_score(y_train, y_pred_train)
print('Accuracy:', Accuracy)


# In[67]:


Confusion_matrix = confusion_matrix(y_train, y_pred_train)
print('Confusion_matrix: \n', Confusion_matrix)


# In[68]:


Classification_report = classification_report(y_train, y_pred_train)
print('Classification_report: \n', Classification_report)


# # Evaluating the testing dataset

# In[69]:


y_pred_test = dtc_model.predict(x_test)


# In[70]:


Accuracy = accuracy_score(y_test, y_pred_test)
print('Accuracy:', Accuracy)


# In[71]:


Confusion_matrix = confusion_matrix(y_test, y_pred_test)
print('Confusion_matrix: \n', Confusion_matrix)


# In[72]:


Confusion_matrix = confusion_matrix(y_test, y_pred_test)
print('Confusion_matrix: \n', Confusion_matrix)


# In[73]:


from sklearn.linear_model import LogisticRegression
model = LogisticRegression()


# In[74]:


model.fit(x_train, y_train)


# In[75]:


y_pred_test = dtc_model.predict(x_test)
y_pred_test


# In[76]:


from sklearn.metrics import accuracy_score, confusion_matrix
confusion_matrix(y_test, y_pred_test)


# In[77]:


accuracy = accuracy_score(y_test, y_pred_test)*100
print('Accuracy of the model is: {:.3f}'.format(accuracy),'%')

