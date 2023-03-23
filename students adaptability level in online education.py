#!/usr/bin/env python
# coding: utf-8

# # Importing Libraries

# In[1]:


import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns


# # Exploratory Data Analysis

# In[2]:


#Data Reading
df= pd.read_csv("students_adaptability_level_online_education.csv")
df.head()


# In[3]:


df.shape


# In[4]:


df.columns


# In[5]:


#Data type:
df.info()


# In[6]:


# checking statistics of dataset:
df.describe().T


# In[ ]:





# In[7]:


# cheking for gaps:
df.isnull().sum()


# # Data Visualisation

# In[8]:


#constructing a general distribution of features by their number
i = 1
plt.figure(figsize = (15,25))
for feature in df:
    plt.subplot(6,3,i)
    sns.countplot(x = feature,data=df)
    i=i+1
    


# In[9]:


#distribution of the number of students depending on the level of their adaptation
i = 1
plt.figure(figsize = (15,25))
for feature in df:
    plt.subplot(6,3,i)
    sns.countplot(x = feature , hue='Adaptivity Level', data = df)
    i +=1


# # Machine learning

# In[29]:


#Encode variables OrdinalEncoder():

from sklearn.preprocessing import OrdinalEncoder
scaler = OrdinalEncoder()
names = df.columns
d = scaler.fit_transform(df)

newdf = pd.DataFrame(d, columns=names)
newdf.head()


# In[30]:


from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score,recall_score
from sklearn.metrics import f1_score
from imblearn.over_sampling import SMOTE


# Split data:

# In[31]:


oversample = SMOTE()
X,y=  oversample.fit_resample(newdf.drop(["Adaptivity Level"],axis=1),newdf["Adaptivity Level"])


# In[32]:


X_train,X_test,y_train,y_test = train_test_split(X,y,test_size = 0.33,random_state = 42)


# In[33]:


print("shape of X_train:",X_train.shape)
print("shape of X_test:",X_test.shape)


# 
# 
# # Model

# Logistic Regression:

# In[41]:


LRC = LogisticRegression(solver="liblinear",max_iter=5000)
LRC.fit(X_train,y_train)
y_pred = LRC.predict(X_test)
print(classification_report(y_test,y_pred))

LRCAcc = accuracy_score(y_pred,y_test)
print('Logistic Regression accuracy is :{:.2f}%'.format(LRCAcc*100))


# K Neighbors:

# In[35]:


knn = KNeighborsClassifier(n_neighbors = 2)
knn.fit(X_train,y_train)
pred=knn.predict(X_test)
print(classification_report(y_test,pred))

knnAcc = accuracy_score(pred,y_test)
print('knn accuracy is :{:.2f}%'.format(knnAcc*100))


# RandomForest Classifier:

# In[36]:


RFC=RandomForestClassifier()
RFC.fit(X_train, y_train)
ypred = RFC.predict(X_test)
print(classification_report(y_test,ypred))

RFCAcc = accuracy_score(ypred,y_test)
print('RandomForest accuracy is :{:.2f}%'.format(RFCAcc*100))


# Confusion matrix:

# In[37]:


from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
cm = confusion_matrix(RFC.predict(X_test),y_test)
disp = ConfusionMatrixDisplay(cm,display_labels=["High","Low","Moderate"])
disp.plot()
plt.title("Confusion Matrix")
plt.show()


# In[ ]:




