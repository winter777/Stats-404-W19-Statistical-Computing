#!/usr/bin/env python
# coding: utf-8

# # Introduction to `pandas`
# References: 
# - "Python for Data Analysis" by Wes McKinney - for discussion on `pandas`
# - ["Hacking Airline DataSet with H2O"](https://github.com/h2oai/h2o-2/wiki/Hacking-Airline-DataSet-with-H2O) -- for Airline data sets of various sizes

# In[1]:


import pandas as pd


# ## Reading-in Data

# In[6]:


df = pd.read_csv(filepath_or_buffer="https://s3.amazonaws.com/h2o-airlines-unpacked/allyears2k.csv",
                 encoding='latin-1')


# In[10]:


# Alternative to ?pd.read_csv to see function arguments:
import inspect
inspect.signature(pd.read_csv)


# ## EDA or Getting to Know your Data Set

# ### Step 1: Read [documentation and data dictionary](http://stat-computing.org/dataexpo/2009/the-data.html)

# ![Screenshot of Airline data documentation](images/Airlines_documentation.png)

# ### Step 2: Basic Checks
# - size
# - shape
# - values
# - `<your favorite check>`

# In[18]:


df.shape


# In[19]:


df.columns


# In[20]:


df.head()


# In[21]:


pd.options.display.max_columns = 50


# In[22]:


df.tail()


# In[34]:


df['Year'].value_counts(sort=False)


# In[37]:


from collections import Counter
Counter(df['Month'])


# From [website](https://github.com/h2oai/h2o-2/wiki/Hacking-Airline-DataSet-with-H2O), this is a `smaller collection of 2000 rows from all years airline data set`.

# In[48]:


df.describe()


# **Caution**: Missing values are ignored and counts not shown.

