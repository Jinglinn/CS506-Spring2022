#!/usr/bin/env python
# coding: utf-8

# In[294]:


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

df = train = pd.read_csv('train.csv')
df.describe()


# In[295]:


def num_nans(df):
    return df.shape[0] - df.dropna().shape[0]
print("there are " +  str(num_nans(df)) + " rows with at least one empty value")


# In[296]:


def drop_na(df):
    return df.dropna(axis = 1, thresh=len(df) - 200)
  
df = drop_na(df)
df.columns


# In[297]:


def to_numerical(df):
    return df['Sex'].apply({'male':0, 'female':1}.get)
df['Sex'] = to_numerical(df)
df.head()


# In[303]:


def extract_names(df):
    df['First Name']= df["Name"].str.split(" ").str.get(2)
    df['Middle Name']= df["Name"].str.split(" ").str.get(3)
    df['Last Name']= df["Name"].str.split(", ").str.get(0)
    df['Title']= df["Name"].str.split(". ").str.get(1)
    df = df[['First Name','Middle Name','Last Name','Title']]
    return df

df[['First Name','Middle Name', 'Last Name','Title']] = extract_names(df)
df.head()


# In[310]:


def replace_with_mean(df):
    return df['Age'].fillna(df['Age'].mean())

df['Age'] = replace_with_mean(df)
df.head(100)


# In[318]:


df.groupby('Survived').Age.mean().plot(kind='bar')
#the average age for not-survied people is around 30 and for survied people is around 28, which is slightly younger


# In[317]:


df.groupby('Sex').Survived.mean().plot(kind='bar')
#female takes a large proportion of the survied population.


# In[319]:


df.groupby('Title').Survived.mean().plot(kind='bar')
#there are some people with tittle that didn't survive. hard to tell the pattern


# In[306]:


train.groupby('Survived').Fare.mean().plot(kind='bar')


# In[70]:


train.groupby('Survived').Fare.mean().plot(kind='box')


# In[71]:


df['Fare']= df['Fare'].sub(df['Fare'].mean())/ df['Fare'].std()
df.head()


# In[320]:



df = df.select_dtypes(include='number')


# In[322]:


def N_most_similar(df, N):
    df = df.drop(['Age','PassengerId'],axis = 1)
    current_pair = np.unravel_index(np.argmin(df.values, axis=None), df.shape)
    current_pair
    return # < your code here >

print("The 3 most similar passengers are: " + str(N_most_similar(df, 3)))


# In[289]:


import requests
import json

"""
    Google Books Api
    See: https://developers.google.com/books/
"""

def get(topic=""):
    BASEURL = 'https://www.googleapis.com/books/v1/volumes'
    headers = {'Content-Type': 'application/json'}
    
    response = requests.get(BASEURL + "?q=" + topic, headers=headers)

    if response.status_code == 200:
        return json.loads(response.content.decode('utf-8'))

    return response

python = get('Python')
data_science = get('data_science')
data_analytics = get('data_analytics')
machine_learning =get('machine_learning')
deep_learning =get('deep_learning')


# In[290]:


python = pd.json_normalize(python['items'] )
data_analytics = pd.json_normalize(data_analytics['items'] )
data_science = pd.json_normalize(data_science['items'] )
machine_learning = pd.json_normalize(machine_learning['items'] )
deep_learning = pd.json_normalize(deep_learning['items'] )

python.to_csv("CSV1.csv",index=False)
data_analytics.to_csv("CSV2.csv",index=False)
data_science.to_csv("CSV3.csv",index=False)
machine_learning.to_csv("CSV4.csv",index=False)
deep_learning.to_csv("CSV5.csv",index=False)


# In[291]:


python= python.rename(columns={"volumeInfo.title": "Title", "volumeInfo.authors": "Authors"})
data_analytics= data_analytics.rename(columns={"volumeInfo.title": "Title", "volumeInfo.authors": "Authors"})
data_science= data_science.rename(columns={"volumeInfo.title": "Title", "volumeInfo.authors": "Authors"})
machine_learning= machine_learning.rename(columns={"volumeInfo.title": "Title", "volumeInfo.authors": "Authors"})
deep_learning= deep_learning.rename(columns={"volumeInfo.title": "Title", "volumeInfo.authors": "Authors"})


# In[292]:


python['Topic'] = 'Python'
data_science['Topic'] = 'data_science'
data_analytics['Topic'] = 'data_analytics'
machine_learning['Topic'] = 'machine_learning'
deep_learning['Topic'] = 'deep_learning'
df1 = pd.concat([python, data_analytics, data_science, machine_learning, deep_learning])
df1.to_csv("CSV.csv",index=False)
print(df1)


# In[281]:


contain_values = df1[df1['Title'].str.contains('Data')]
print(contain_values)


# In[293]:


df1['Authors'] = df1['Authors'].str.get(0)
start_values  = df1[df1['Authors'].str.match('E')]
print(start_values)


# In[ ]:





# In[ ]:




