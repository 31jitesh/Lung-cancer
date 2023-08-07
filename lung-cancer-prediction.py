#!/usr/bin/env python
# coding: utf-8

# # Import Libraries

# In[1]:


get_ipython().system('pip install dtreeviz')


# In[4]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, ExtraTreesClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier
from sklearn.metrics import precision_score, recall_score, accuracy_score, f1_score, confusion_matrix, ConfusionMatrixDisplay, classification_report



import warnings
warnings.filterwarnings("ignore")


# # Load Data

# In[5]:


df = pd.read_csv(r"C:\Users\Hp\Downloads\cancer patient data sets.csv")
df


# # Data Cleaning & Visualization

# In[4]:


df.isnull().sum()


# In[5]:


sns.heatmap(df.isnull(), cmap = 'viridis')


# In[6]:


df.drop(columns=['index', 'Patient Id'], axis=1, inplace=True)
df


# In[7]:


df.size


# In[8]:


df.dtypes


# In[9]:


df.iloc[:, 1:24].plot(title="Dataset Details")


# In[10]:


df_corr = df.corr()
df_corr


# In[11]:


plt.title("Correlation Matrix")
sns.heatmap(df_corr, cmap='viridis')


# In[12]:


sea = sns.FacetGrid(df, col = "Level", height = 4)
sea.map(sns.distplot, "Age")


# In[13]:


sea = sns.FacetGrid(df, col = "Level", height = 4)
sea.map(sns.distplot, "Gender")


# In[14]:


x = df.iloc[:, 0:23]
x


# In[15]:


x = x.values
x


# In[16]:


df['Level'].unique()


# In[17]:


df['Level'].value_counts()


# In[18]:


df['Level'].replace(to_replace = 'Low', value = 0, inplace = True)
df['Level'].replace(to_replace = 'Medium', value = 1, inplace = True)
df['Level'].replace(to_replace = 'High', value = 2, inplace = True)

df['Level'].value_counts()


# In[19]:


plt.figure(figsize = (20, 27))

for i in range(24):
    plt.subplot(8, 3, i+1)
    sns.distplot(df.iloc[:, i], color = 'red')
    plt.grid()


# In[20]:


plt.figure(figsize = (11, 9))
plt.title("Lung Cancer Chances Due to Air Polution")
plt.pie(df['Level'].value_counts(), explode = (0.1, 0.02, 0.02), labels = ['High', 'Medium', 'Low'], autopct = "%1.2f%%", shadow = True)
plt.legend(title = "Lung Cancer Chances", loc = "lower left")


# In[21]:


sns.displot(df['Level'], kde=True)


# In[22]:


y = df.Level.values
y


# # Train & Test Splitting the Data

# In[23]:


x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=1)


# # Function of Measure Performance

# In[24]:


def perform(y_pred):
    print("Precision : ", precision_score(y_test, y_pred, average = 'micro'))
    print("Recall : ", recall_score(y_test, y_pred, average = 'micro'))
    print("Accuracy : ", accuracy_score(y_test, y_pred))
    print("F1 Score : ", f1_score(y_test, y_pred, average = 'micro'))
    cm = confusion_matrix(y_test, y_pred)
    print("\n", cm)
    print("\n")
    print("**"*27 + "\n" + " "* 16 + "Classification Report\n" + "**"*27)
    print(classification_report(y_test, y_pred))
    print("**"*27+"\n")
    
    cm = ConfusionMatrixDisplay(confusion_matrix = cm, display_labels=['Low', 'Medium', 'High'])
    cm.plot()


# # Random Forest

# In[25]:


model_rf = RandomForestClassifier()
model_rf.fit(x_train, y_train)


# In[26]:


y_pred_rf = model_rf.predict(x_test)


# In[27]:


perform(y_pred_rf)


# ## Saving the Random Forest Model

# In[28]:


filename = 'Lung_Cancer_RF.h5'
pickle.dump(model_rf, open(filename, 'wb'))
print("Model Saved.")


# # ADABoost Classifier

# In[29]:


model_ada = AdaBoostClassifier()
model_ada.fit(x_train, y_train)


# In[30]:


y_pred_ada = model_ada.predict(x_test)


# In[31]:


perform(y_pred_ada)


# ## Saving the ADABoost Classifier Model

# In[32]:


filename = 'Lung_Cancer_ADA.h5'
pickle.dump(model_ada, open(filename, 'wb'))
print("Model Saved.")


# # Extra Trees Classifier

# In[33]:


model_etc = ExtraTreesClassifier()
model_etc.fit(x_train, y_train)


# In[34]:


y_pred_etc = model_etc.predict(x_test)


# In[35]:


perform(y_pred_etc)


# ## Saving the Extra Trees Classifier Model

# In[36]:


filename = 'Lung_Cancer_ETC.h5'
pickle.dump(model_etc, open(filename, 'wb'))
print("Model Saved.")


# # Decision Tree

# In[37]:


model_dt = DecisionTreeClassifier()
model_dt.fit(x_train, y_train)


# In[38]:


y_pred_dt = model_dt.predict(x_test)


# In[39]:


perform(y_pred_dt)


# ## Saving the Decision Tree Model

# In[40]:


filename = 'Lung_Cancer_DT.h5'
pickle.dump(model_dt, open(filename, 'wb'))
print("Model Saved.")


# ## Decision Tree Visualization

# In[41]:


feature_names = df.columns[0:23]
viz = df.copy()
viz["Level"]=viz["Level"].values.astype(str)
print(viz.dtypes)
target_names = viz['Level'].unique().tolist()


# In[42]:


from sklearn.tree import plot_tree # tree diagram

plt.figure(figsize=(25, 20))
plot_tree(model_dt, feature_names = feature_names, class_names = target_names, filled = True, rounded = False)

plt.savefig('tree_visualization.png')


# In[43]:


import dtreeviz

viz_model = dtreeviz.model(model_dt,
                           X_train=x_train, y_train=y_train,
                           feature_names=feature_names,
                           target_name='Lung Cancer',
                           class_names=['Low', 'Medium', 'High'])

v = viz_model.view()     # render as SVG into internal object
v.save("Lung Cancer.svg")  # save as svg


# In[44]:


viz_model.view()


# # Logistic Regression

# In[45]:


model_lr = LogisticRegression()
model_lr.fit(x_train, y_train)


# In[46]:


y_pred_lr = model_lr.predict(x_test)


# In[47]:


perform(y_pred_lr)


# ## Saving the Logistic Regression Model

# In[48]:


filename = 'Lung_Cancer_LR.h5'
pickle.dump(model_lr, open(filename, 'wb'))
print("Model Saved.")


# # XGBoost Classifier

# In[49]:


model_xgb = XGBClassifier()
model_xgb.fit(x_train, y_train)


# In[50]:


y_pred_xgb = model_xgb.predict(x_test)


# In[51]:


perform(y_pred_xgb)


# ## Saving the XGBoost Classifier Model

# In[52]:


filename = 'Lung_Cancer_XGB.h5'
pickle.dump(model_xgb, open(filename, 'wb'))
print("Model Saved.")


# # Multi-Layer Perceptron Classifier

# In[53]:


model_mlp = MLPClassifier()
model_mlp.fit(x_train, y_train)


# In[54]:


y_pred_mlp = model_mlp.predict(x_test)


# In[55]:


perform(y_pred_mlp)


# ## Saving the Multi-Layer Perceptron Classifier Model

# In[56]:


filename = 'Lung_Cancer_MLP.h5'
pickle.dump(model_mlp, open(filename, 'wb'))
print("Model Saved.")


# <br>
# <p style="text-align:center; font-weight:800; font-size:18px"><em>Thank you For viewing this Notebook 😃, do upvote 🔼 if you like it and please feel free to provide any feedback.</em>
# <p style="text-align:center"><img src="https://i.pinimg.com/originals/40/12/1a/40121a3616ecf2439a5b04d733b6f437.gif" width="480" height="200"></p>
