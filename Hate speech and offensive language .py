#!/usr/bin/env python
# coding: utf-8

# To detect the hate speech from the text data

# In[1]:


import numpy as np
import pandas as pd
import os
os.getcwd()


# In[3]:


import matplotlib.pyplot as plt


# In[4]:


from wordcloud import WordCloud


# In[5]:


from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder


# In[6]:


import re


# In[7]:


import nltk


# In[8]:


nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import CountVectorizer


# In[9]:


#train split and fit models
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB


# In[10]:


from sklearn.metrics import confusion_matrix, accuracy_score


# In[11]:


dataset=pd.read_csv("labeled_data.csv")


# In[12]:


dataset.head()


# In[51]:


dataset['tweet'][6]
                


# In[13]:


dataset.info()


# In[14]:


dataset.isnull().sum()


# In[15]:


dataset.columns


# In[17]:


dataset.describe().T


# In[18]:


dt_trasformed = dataset[['class', 'tweet']]
y = dt_trasformed.iloc[:, :-1].values


# In[19]:


dataset.head()


# In[21]:


y = np.array(ct.fit_transform(y))


# In[20]:


ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [0])], remainder='passthrough')


# In[22]:


print(y)


# In[60]:


X_train


# In[61]:


X_test


# In[ ]:


##I separated this y in two variables that we will use to fit hate speech models and offensive speech models


# In[63]:


y_test


# In[62]:


y_train


# In[23]:


y_df = pd.DataFrame(y)
y_hate = np.array(y_df[0])
y_offensive = np.array(y_df[1])


# In[ ]:





# In[24]:


print(y_hate)
print(y_offensive)


# In[ ]:


##cleaning the texts


# In[25]:


corpus = []
for i in range(0, 24783):
  review = re.sub('[^a-zA-Z]', ' ', dt_trasformed['tweet'][i])
  review = review.lower()
  review = review.split()
  ps = PorterStemmer()
  all_stopwords = stopwords.words('english')
  all_stopwords.remove('not')
  review = [ps.stem(word) for word in review if not word in set(all_stopwords)]
  review = ' '.join(review)
  corpus.append(review)


# In[26]:


cv = CountVectorizer(max_features = 2000)


# In[27]:


X = cv.fit_transform(corpus).toarray()


# In[28]:


X_train, X_test, y_train, y_test = train_test_split(X, y_hate, test_size = 0.20, random_state = 0)


# In[64]:


y_hate


# In[65]:


X


# In[29]:


##Finding the best models to predict hate speech¶


# In[ ]:


#Naive Bayes


# In[30]:


classifier_np = GaussianNB()


# In[31]:


classifier_np.fit(X_train, y_train)


# In[ ]:


##Decision Tree


# In[32]:


classifier_dt = DecisionTreeClassifier(criterion = 'entropy', random_state = 0)
classifier_dt.fit(X_train, y_train)


# In[ ]:


##knn


# In[33]:


classifier_knn = KNeighborsClassifier(n_neighbors = 5, metric = 'minkowski', p = 2)
classifier_knn.fit(X_train, y_train)


# In[ ]:


##logistic regression


# In[34]:


classifier_lr = LogisticRegression(random_state = 0)
classifier_lr.fit(X_train, y_train)


# In[ ]:


##random forest


# In[35]:


classifier_rf = RandomForestClassifier(n_estimators = 10, criterion = 'entropy', random_state = 0)
classifier_rf.fit(X_train, y_train)


# In[ ]:


##Making the Confusion Matrix for each model¶


# In[36]:


#Naive Bayes
y_pred_np = classifier_np.predict(X_test)
cm = confusion_matrix(y_test, y_pred_np)
print(cm)


# In[37]:


##Decision Tree
y_pred_dt = classifier_dt.predict(X_test)
cm = confusion_matrix(y_test, y_pred_dt)
print(cm)


# In[38]:


##Linear Regression
y_pred_lr = classifier_lr.predict(X_test)
cm = confusion_matrix(y_test, y_pred_lr)
print(cm)


# In[39]:


###Random Florest
y_pred_rf = classifier_rf.predict(X_test)
cm = confusion_matrix(y_test, y_pred_rf)
print(cm)


# In[40]:


rf_score = accuracy_score(y_test, y_pred_rf)
lr_score = accuracy_score(y_test, y_pred_lr)
dt_score = accuracy_score(y_test, y_pred_dt)
np_score = accuracy_score(y_test, y_pred_np)


# In[41]:



print('Random Forest Accuracy: ', str(rf_score))
print('Linear Regression Accuracy: ', str(lr_score))
print('Decision Tree Accuracy: ', str(dt_score))
print('Naive Bayes Accuracy: ', str(np_score))


# ##So, Linear Regression looks better to predict hate speech based on this dataset. It's important to emphasize Random Forest and Decision Tree had great results as well. This Dataset looks like a product of artificial intelligence to classify hate speech and offensive terms. What I think way more technological than these models that I implemented here. It was just a study of the implementation of NLP based on the bag of words method.##
