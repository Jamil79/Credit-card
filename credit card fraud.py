#!/usr/bin/env python
# coding: utf-8

# In[7]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error


# In[8]:


df = pd.read_csv('C:/Users/96132/Desktop/creditcard.csv')
df.shape


# In[4]:


df.head()


# In[ ]:


## supervised linear regression model


# In[11]:


#Split into explanatory and response variables
X = df[['V1', 'V2', 'V3', 'V4', 'V5', 'V6', 'V7', 'V8', 'V9', 'V10', 'V11', 'V12']]
y = df['Amount']

#Split into train and test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = .30, random_state=42) 

lm_model = LinearRegression(normalize=True) # Instantiate
lm_model.fit(X_train, y_train) #Fit
        
#Predict and score the model
y_test_preds = lm_model.predict(X_test) 
"The r-squared score for your model was {} on {} values.".format(r2_score(y_test, y_test_preds), len(y_test))


# In[ ]:


## prediction of the model


# In[12]:


y_test_preds


# In[ ]:


## statistics of the model


# In[10]:


df1 = df[['V1', 'V2', 'V3', 'V4', 'Amount']]
df1.describe()


# In[ ]:


## correlation of the model


# In[14]:


corr = df1.corr(method='pearson')
corr


# In[ ]:


## graph of the fraudulent


# In[21]:


amount = df['Amount']
graph = amount.plot(title='Fraud')
graph


# In[ ]:




