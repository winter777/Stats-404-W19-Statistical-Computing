#!/usr/bin/env python
# coding: utf-8

# # Iterative Model Development Steps with Application to Airlines Dataset
# - Steps are outlined in https://goo.gl/A7P4vX
# - Link to airlines [dataset](https://github.com/h2oai/h2o-2/wiki/Hacking-Airline-DataSet-with-H2O)

# In[82]:


from collections import Counter
import inspect
from joblib import dump, load
import numpy as np
import pandas as pd
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression


# In[48]:


pd.options.display.max_columns = 50
pd.options.display.max_rows = 8


# ## Step 1: Understand Different Modeling Approaches
# Please see [https://goo.gl/A7P4vX](https://goo.gl/A7P4vX)

# ## Step 2: Understand Business Use Case
# - Client: Airline
# - Statement of Problem: Airline has to compensate passangers if flight was delayed by 2+ hours or if flight arrived 3+ hours later.
# - Question: Are there (any) aspects of delay that could have been prevented?

# - Production environment:
#   - Jupyter notebook (for POC)
#   - Running Python 3.7
#   - requirements.txt

# - Outcome variable: (arrival delay of 3+ hours or departure delay of 2+ hours) or not, per https://upgradedpoints.com/flight-delay-cancelation-compensation

# ## Step 3: Get Access to Data

# ### Step 3-a: Read-in Data

# In[3]:


file_name = "https://s3.amazonaws.com/h2o-airlines-unpacked/year2012.csv"
df = pd.read_csv(filepath_or_buffer=file_name,
                 encoding='latin-1')
# df = pd.read_csv("2012.csv")


# ### Step 3-b: EDA of Data

# In[4]:


df.shape


# In[5]:


Counter(df['Month'])


# In[6]:


df.head()


# In[7]:


min(df['DepTime']), max(df['DepTime'])


# In[8]:


min(df['ArrTime']), max(df['ArrTime'])


# In[46]:


Counter(df['UniqueCarrier'])


# ## Step 4: To come a little later...

# ## Step 5: Feature Engineering for Baseline Model (v0)

# - What are potential features we want to include in model?

# ### Step 5-a: Create an Outcome Variable

# In[23]:


# Similar to function introduced in Class 2:

def delays_requiring_compensation(arrival_delay, departure_delay):
    """Fcn to return if arrival and/or departure delay resulted in passenger
       compensation.
       
       Arguments:
           - arrival_delay:   delay in minutes
           - departure_delay: delay in minutes
       
       Returns:
           - number of delays (arrival and or departure) that were delayed
             so long that passenger got compensated
    """
    count = 0
    if (arrival_delay/60.0 >= 3) | (departure_delay/60.0 >= 2):
        # If arrival delay is 3+ hours, or if departure delay is 2+ hours:
        count += 1
    return count


# In[15]:


df['compensated_delays'] = df[['ArrDelay', 'DepDelay']].apply(
    lambda row: delays_requiring_compensation(row[0], row[1]),
    axis=1)
df[['ArrDelay', 'DepDelay', 'compensated_delays']].head()


# In[24]:


Counter(df['compensated_delays'])


# Let's stop and think about this number... What are client implications?

# ### Step 5-b: Create a Time-of-Day Variable
# - Per [documentation](http://stat-computing.org/dataexpo/2009/the-data.html) and EDA, time of day is recorded in minutes (float).

# In[25]:


print(df['DepTime'][0])
str(int(df['DepTime'][0])).zfill(4)


# In[26]:


print(min(df['DepTime']))
str(int(min(df['DepTime']))).zfill(4)


# In[27]:


# There are missing departure times:
df['DepTime'] = df['DepTime'].fillna(9999.0)


# In[55]:


df['Dep_Hour'] = df['DepTime'].apply(lambda x:
                                     int(
                                         str(int(x)).zfill(4)[0:2]
                                     ))


# In[56]:
# What would be a good comment to the code above?


df['Dep_Hour'].value_counts(sort=False)


# Does anything look weird?

# In[57]:


index_24 = np.where(df['Dep_Hour'] == 24)
df['Dep_Hour'].iloc[index_24] = 0


# In[58]:


df['Dep_Hour'].value_counts(sort=False)


# ### Step 3-c: Create Indicator Variables from Features for Use with Sklearn
# Features:
# - Month
# - Day of Week
# - Time of Day

# In[65]:


features_tod = pd.get_dummies(df['Dep_Hour'], drop_first=True, prefix="tod_")


# In[66]:


features_tod.head()


# In[67]:


features_month = pd.get_dummies(df['Month'], drop_first=True, prefix="mo_")


# In[68]:


features_dow = pd.get_dummies(df['DayOfWeek'], drop_first=True, prefix="dow_")


# In[69]:


features = pd.concat([features_tod, features_month, features_dow],
                     axis=1,
                     join='inner')


# In[70]:


features.columns


# In[75]:


dataset = pd.concat([features, df['compensated_delays']],
                     axis=1)


# In[76]:


dataset.shape


# ## Step 4: Determine Data Splits
# What are some data splits that you would propose?

# In[77]:


df_tmp, df_test = train_test_split(dataset,
                                   test_size=0.25,
                                   random_state=2019,
                                   stratify=dataset['compensated_delays'])


# In[78]:


df_train, df_valid = train_test_split(df_tmp,
                                      test_size=0.25,
                                      random_state=2019,
                                      stratify=df_tmp['compensated_delays'])


# In[79]:


df_train['compensated_delays'].value_counts(sort=False)


# In[80]:


df_valid['compensated_delays'].value_counts(sort=False)


# In[81]:


df_test['compensated_delays'].value_counts(sort=False)


# ## Step 6: Estimate a Baseline Model (v0)

# In[83]:


y = df_train['compensated_delays']
X = df_train.drop(columns=['compensated_delays'])


# In[84]:


inspect.signature(LogisticRegression)


# In[103]:


est_model = LogisticRegression(penalty="l2",
                               C=0.5,
                               fit_intercept=True,
                               class_weight='balanced',
                               random_state=2019,
                               max_iter=10000,
                               solver='lbfgs')


# In[104]:


est_model


# In[ ]:


est_model.fit(X, y)


# In[88]:


# Save model, per: https://scikit-learn.org/stable/modules/model_persistence.html
dump(est_model, 'logistic.joblib')

## Load saved model:
# est_model = load('logistic.joblib') 


# - Aside: creating a filename that includes a [timestamp](https://stackoverflow.com/questions/10607688/how-to-create-a-file-name-with-the-current-date-time-in-python)
# - Logistic Regression with [Cross-Validation](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegressionCV.html)
# - (in general) Cross-Validation with sklearn: [approach 1](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html) and [approach 2](https://scikit-learn.org/0.16/modules/generated/sklearn.grid_search.GridSearchCV.html)

# ## Step 7: Interpret Results

# In[100]:
# DataFrame of feature coefficients:
coefficients = [round(x, 2) for x in est_model.coef_.tolist()[0]]
coef_df = pd.concat([pd.DataFrame(X.columns),
                     pd.DataFrame(coefficients)],
                    axis = 1)
coef_df.columns = ["feature_name", "coef_est"]

# DataFrame of intercept coefficient:
intercept_df = pd.DataFrame(["intercept", est_model.intercept_.tolist()[0]]).T
intercept_df.columns = ["feature_name", "coef_est"]

# Combined DataFrame:
coef_df = coef_df.append(intercept_df)
coef_df.head()




# In[ ]:
# Compute odds:
coef_df["odds"] = [round(np.exp(x), 2) for x in coef_df['coef_est']]




# In[ ]:
# Compute probabilities, which need intercept:
intercept = intercept_df['coef_est'].values.tolist()[0]
intercept


def inverse_logit(intercept, coefficient):
    """Fcn to help calculate probability associated with flight delay,
       given our variable.
    """
    if coefficient == intercept:
        # Make sure we don't double-count the intercept:
        coefficient = 0
    inv_logit = np.exp(intercept + coefficient)/(1+np.exp(intercept + coefficient))
    return round(inv_logit, 2)
                           
coef_df["prob_delay"] = [inverse_logit(intercept, x) for x in coef_df['coef_est']]



coef_df.T







# ## Step 8: Evaluate Performance

# In[ ]:





# ## Step 9: Determine Next Steps
