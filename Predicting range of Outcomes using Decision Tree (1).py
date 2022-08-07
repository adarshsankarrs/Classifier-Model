#!/usr/bin/env python
# coding: utf-8

# In[4]:


pip install -U scikit-learn


# In[5]:


#Loading required libraries


# In[6]:


import pandas as pd


# In[7]:


from sklearn.tree import DecisionTreeClassifier


# In[8]:


from sklearn.model_selection import train_test_split


# In[9]:


from sklearn import metrics


# In[10]:


# Load Dataset


# In[11]:


column_names = ['pregnant', 'glucose', 'bp', 'skin', 'insulin', 'bmi', 'pedigree', 'age', 'label']


# In[12]:


df = pd.read_csv(r'C:\Users\HP\Desktop\Python\diabetes.csv', header=0, names=column_names) 


# In[13]:


df


# In[14]:


#Need to split dataset into features and target variable


# In[15]:


feature_columns = ['pregnant', 'insulin', 'bmi', 'age','glucose','bp','pedigree']


# In[16]:


X = df[feature_columns] # features


# In[17]:


Y = df.label # target variable


# In[18]:


# dataset into training set and testing set


# In[19]:


X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=1) # around 70% training set and 30% testing set


# In[20]:


# build decision tree


# In[21]:


clf = DecisionTreeClassifier() #classifier object


# In[22]:


# We train the classifier


# In[23]:


clf = clf.fit(X_train,Y_train)


# In[24]:


#Looking to predict the response for this particular dataset


# In[25]:


Y_pre = clf.predict(X_test)


# In[26]:


#Visualizing Decision Tree


# In[27]:


#Installing graphviz and pydotplus


# In[28]:


pip install graphviz


# In[29]:


conda install graphviz


# In[30]:


pip install pydotplus


# In[31]:


from sklearn.tree import export_graphviz


# In[39]:


get_ipython().system('pip install --upgrade scikit-learn==0.20.3')


# In[40]:


from sklearn.externals.six import StringIO  


# In[41]:


from IPython.display import Image  


# In[42]:


import pydotplus


# In[43]:


dot_data = StringIO()


# In[44]:


export_graphviz(clf, out_file=dot_data, filled=True, rounded=True, special_characters=True,feature_names = feature_columns,class_names=['0','1'])


# In[45]:


graph = pydotplus.graph_from_dot_data(dot_data.getvalue())


# In[46]:


graph.write_png('diabetes.png')


# In[47]:


Image(graph.create_png())


# In[48]:


#Evaluating accuracy of the model


# In[50]:


print("Accuracy:",metrics.accuracy_score(Y_test, Y_pre))


# In[51]:


#Tuned tree


# In[52]:


# Optimizing features


# In[53]:


clf = DecisionTreeClassifier(criterion="entropy", max_depth=3)


# In[54]:


clf = clf.fit(X_train,Y_train)


# In[55]:


Y_pred = clf.predict(X_test)


# In[56]:


# Visualizing tuned tree


# In[57]:


from IPython.display import Image


# In[58]:


from sklearn.tree import export_graphviz


# In[59]:


import pydotplus


# In[60]:


dot_data = StringIO()


# In[62]:


export_graphviz(clf, out_file=dot_data, filled=True, rounded=True,special_characters=True, feature_names = feature_columns,class_names=['0','1'])


# In[63]:


graph = pydotplus.graph_from_dot_data(dot_data.getvalue())


# In[64]:


graph.write_png('diabetes.png')


# In[65]:


Image(graph.create_png())


# In[66]:


# Accuracy after optimization of features


# In[68]:


print("Accuracy:",metrics.accuracy_score(Y_test, Y_pred))


# In[ ]:




