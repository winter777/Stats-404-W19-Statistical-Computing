#!/usr/bin/env python
# coding: utf-8

# # Introduction to `pandas`
# References: 
# - "Python for Data Analysis" by Wes McKinney - for discussion on `pandas`
# - ["Hacking Airline DataSet with H2O"](https://github.com/h2oai/h2o-2/wiki/Hacking-Airline-DataSet-with-H2O) -- for Airline data sets of [various sizes](https://s3.amazonaws.com/h2o-airlines-unpacked/)

# In[1]:
# **Prerequisities**: `pandas` is installed in your virtual environment


import pandas as pd


# ## `pandas` Data Types

# ### `Series`

tmp = pd.Series([3, 5, -1])
tmp

tmp.values
tmp.index


# `DataFrame`
# In[6]:


df_tmp = pd.DataFrame([3, 5, -1], columns=['col_name'])
df_tmp


df_tmp.shape
df_tmp.columns
df_tmp['col_name'].values
# In[10]:


df_tmp.index
# ## Reading-in Data
file_name = "https://s3.amazonaws.com/h2o-airlines-unpacked/year1987.csv"
df = pd.read_csv(filepath_or_buffer=file_name,
                 encoding='latin-1',
                 nrows=1000
                )

# Alternative to ?pd.read_csv to see function arguments:
import inspect
inspect.signature(pd.read_csv)


# Other file formats you can read-in:
# - CSV and non-CSV delimited: `read_table`
# - JSON via `read_json`
# - fixed-width format via `read_fwf`
# - data on clipboard via `read_clipboard`

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



# In[48]:


df.describe()


# **Caution**: Missing values are ignored and counts not shown.

df.info()
# ## Aside: Missing Values in Python
# When you need to check for missing values -- similar to R, you can't do `variable == NaN' -- you can check for:
# - `variable_value is None`
# - `math.isnan(variable_value)` -- after importing the `math` library
# - `pd.isnull(variable_value)`
# - non-emptiness per section [Truth Value testing](https://docs.python.org/3/library/stdtypes.html): 
#   - `not []`
#   - `not ""`
#   - etc.
