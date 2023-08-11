#!/usr/bin/env python
# coding: utf-8

# In[3]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[4]:


import sklearn


# In[5]:


df = pd.read_csv('Hospital.csv')


# In[6]:


df.head()


# In[7]:


df.describe()


# In[8]:


df.isnull().sum()
#here we found that the most null values lies in BMI, neutrophils, basophils, Lymphocyte, Creatine kinase,PH, Biocarbonate, lactic acid, PCO2 


# In[9]:


df.info()
#we fount all null data that we have mentioned earlier is float
#so we basically have to manage float null values


# In[10]:


from sklearn.impute import SimpleImputer
imputer = SimpleImputer(missing_values=np.NaN, strategy='mean')


# In[11]:


float_column = df.select_dtypes(include ='float64').columns


# In[12]:


imputer.fit(df[float_column])


# In[13]:


df[float_column] = imputer.transform(df[float_column])
df = df.dropna()


# In[14]:


df.isna().sum()


# In[15]:


x = df.drop(columns ='outcome')
y = df['outcome']
y = y.astype({'outcome':'int'})
y


# In[16]:


fig, ax = plt.subplots(figsize =(8,5), dpi = 100)
patches, texts, autotexts = ax. pie(df['outcome'].value_counts(), autopct= '%1.1f%%', shadow = True, startangle = 90, explode=(0.1,0, 0), labels = ['Alive', 'Death', 'outlier'])
plt.title('Outcome Distribution', fontsize = 15)


# In[17]:


import plotly.express as px
fig = px.histogram(df,x ='age', color ='outcome', marginal = 'box', hover_data = df.columns)
fig.show()


# In[18]:


fig = px.histogram(df,x ='BMI', color ='outcome', marginal = 'box', hover_data = df.columns)
fig.show()


# In[19]:


fig = px.histogram(df,x ='SP O2', color ='outcome', marginal = 'box', hover_data = df.columns)
fig.show()


# In[20]:


fig = px.histogram(df,x ='heart rate', color ='outcome', marginal = 'box', hover_data = df.columns)
fig.show()


# In[21]:


df['gendera'].value_counts()


# In[22]:


plt.figure(figsize=(12,8))
plot = sns.countplot(df['gendera'], hue = df['outcome'])
plt.xlabel('Gender', fontsize = 13, weight ='bold')
plt.ylabel('Count', fontsize = 13, weight ='bold')
plt.xticks(np.arange(2), ['Male', 'Female'], rotation ='vertical', weight ='bold')

for i in plot.patches:
    plot.annotate(format(i.get_height()),
                 (i.get_x()+ i.get_width()/2,
                 i.get_height()), ha ='center', va ='center', 
                  size = 10, xytext =(0,8),
                  textcoords ='offset points')
plt.show()
    


# In[23]:


col = ['group', 'gendera', 'hypertensive', 'atrialfibrillation', 'CHD with no MI', 'diabetes', 'deficiencyanemias', 'depression', 'Hyperlipemia', 'Renal failure', 'COPD', 'outcome']


# In[24]:


corr = df[col].corr()


# In[25]:


plt.figure(figsize =(12,8))
sns.heatmap(corr, annot=True, cmap='PuBuGn')


# In[26]:


plt.figure(figsize=(9,4))
df['age'].plot(kind='kde')


# In[27]:


plt.figure(figsize=(9,4))
df['EF'].plot(kind='kde')


# In[28]:


plt.figure(figsize=(9,4))
df['RBC'].plot(kind='kde')


# In[29]:


plt.figure(figsize=(9,4))
df['Creatinine'].plot(kind='kde')


# In[30]:


plt.figure(figsize=(9,4))
df['Blood calcium'].plot(kind='kde')


# In[31]:


from sklearn.preprocessing import StandardScaler


# In[32]:


scale = StandardScaler()


# In[33]:


scaled = scale.fit_transform(x)


# In[34]:


scaled


# In[35]:


final_x = pd.DataFrame(scaled, columns = x.columns)


# In[36]:


final_x.head()


# In[37]:


y.head()


# In[38]:


from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x,y, test_size=0.3, random_state=123)


# In[39]:


print(x_train.shape, x_test.shape)


# In[40]:


x_train.drop(columns ='ID', inplace=True)
x_test.drop(columns ='ID', inplace=True)


# In[41]:


x_train.head()


# In[43]:


get_ipython().system('pip install xgboost')


# In[44]:


from xgboost import XGBClassifier, plot_tree, plot_importance


# In[45]:


xgb = XGBClassifier(random_state=42)
xgb.fit(x_train, y_train)


# In[46]:


pred = xgb.predict(x_test)


# In[47]:


pred


# In[53]:


from sklearn.metrics import classification_report, accuracy_score, confusion_matrix


# In[54]:


cf = confusion_matrix(y_test, pred)


# In[55]:


cf


# In[56]:


print(classification_report(y_test, pred))


# In[57]:


#Now compare the value
combine = np.concatenate((y_test.values.reshape(len(y_test),1), pred.reshape(len(pred),1)),1)


# In[60]:


combine_result = pd.DataFrame(combine, columns =['y_test', 'y_pred'])


# In[61]:


combine_result


# In[62]:


from sklearn.metrics import plot_roc_curve, plot_precision_recall_curve


# In[ ]:


plot_roc_curve(xgb, x_test, y_test)
plt.plot([0,1],[0,1], color='brown', ls='-')

