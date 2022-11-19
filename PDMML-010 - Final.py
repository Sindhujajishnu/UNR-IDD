#!/usr/bin/env python
# coding: utf-8

# ###  Importing libraries

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import scipy.stats as stats
sns.color_palette()
from sklearn.metrics import classification_report
import warnings
warnings.filterwarnings('ignore')
pd.options.display.float_format = "{:.2f}".format
# Label encoding target feature
from sklearn.model_selection import train_test_split

import seaborn as sn
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_selection import SelectKBest
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn import preprocessing


# ### Loading Dataset

# In[2]:


UNR_df = pd.read_csv('UNR-IDD.csv')


# In[3]:


UNR_df.head(5)


# In[4]:


UNR_df.info()


# In[5]:


# Checking for duplicates
UNR_df[UNR_df.duplicated() == True]


# In[6]:


# Missing values checking
UNR_df.isna().sum()


# In[7]:


# Converting object to categorical features
for i in UNR_df:
    if UNR_df[i].dtypes == 'object':
        UNR_df[i] = UNR_df[i].astype('category')


# In[8]:


# Numerical columns
num_cols = list(UNR_df._get_numeric_data().columns)
num_cols


# In[9]:


# Categorical columns
cat_cols = ['Label', 'Binary Label', 'Switch ID', 'Port Number']


# In[10]:


# Printing unique values
for i in cat_cols:
    print(f'{i}: {UNR_df[i].nunique()}')


# In[11]:


UNR_df.describe().T


# ## Exploratory Data Analysis

# - Function: plotting numerical features

# In[12]:


def histogram_boxplot(feature, figsize=(10,7), bins = None):
    """ Boxplot and histogram combined
    feature: 1-d feature array
    figsize: size of fig (default (9,8))
    bins: number of bins (default None / auto)
    """
    f2, (ax_box2, ax_hist2) = plt.subplots(nrows = 2, # Number of rows of the subplot grid= 2
                                           sharex = True, # x-axis will be shared among all subplots
                                           gridspec_kw = {"height_ratios": (.25, .75)}, 
                                           figsize = figsize 
                                           ) # creating the 2 subplots
    sns.boxplot(feature, ax=ax_box2, showmeans=True, color='skyblue') # boxplot will be created and a star will indicate the mean value of the column
    sns.distplot(feature, kde=F, ax=ax_hist2, bins=bins,color = 'green') if bins else sns.distplot(feature, kde=False, ax=ax_hist2) # For histogram
    ax_hist2.axvline(np.mean(feature), color='blue', linestyle='--') # Add mean to the histogram
    ax_hist2.axvline(np.median(feature), color='black', linestyle='-') # Add median to the histogram


# ### Univariate analysis (continuous numerical)

# In[13]:


continuous_num_cols = ['Received Packets', 'Received Bytes',  'Sent Bytes',
            'Sent Packets', 'Port alive Duration (S)', 'Delta Received Packets',
            'Delta Received Bytes', 'Delta Sent Packets', 'Delta Sent Bytes', 'Delta Port alive Duration (S)',
            'Connection Point', 'Total Load/Rate', 'Total Load/Latest', 'Unknown Load/Rate',
            'Unknown Load/Latest', 'Latest bytes counter', 'Active Flow Entries',
            'Packets Looked Up', 'Packets Matched']


# In[14]:


# Mean, Median and Mode
for i in continuous_num_cols:
    mean=UNR_df[i].mean()
    median=UNR_df[i].median()
    mode=UNR_df[i].tolist()[0]
    print(f'Feature: {i}, Mean: {mean}, Median: {median}, Mode: {mode}')


# In[15]:


# Numerical continous analysis
histogram_boxplot(UNR_df['Received Packets'])


# In[16]:


histogram_boxplot(UNR_df['Received Bytes'])


# In[17]:


histogram_boxplot(UNR_df['Sent Packets'])


# In[18]:


histogram_boxplot(UNR_df['Sent Bytes'])


# In[19]:


histogram_boxplot(UNR_df['Port alive Duration (S)'])


# In[20]:


histogram_boxplot(UNR_df['Delta Received Packets'])


# In[21]:


histogram_boxplot(UNR_df['Delta Received Bytes'])


# In[22]:


histogram_boxplot(UNR_df['Delta Sent Packets'])


# In[23]:


histogram_boxplot(UNR_df['Delta Sent Bytes'])


# In[24]:


histogram_boxplot(UNR_df['Delta Port alive Duration (S)'])


# In[25]:


histogram_boxplot(UNR_df['Connection Point'])


# In[26]:


histogram_boxplot(UNR_df['Total Load/Rate'])


# In[27]:


histogram_boxplot(UNR_df['Total Load/Latest'])


# In[28]:


histogram_boxplot(UNR_df['Unknown Load/Rate'])


# In[29]:


histogram_boxplot(UNR_df['Unknown Load/Latest'])


# In[30]:


histogram_boxplot(UNR_df['Latest bytes counter'])


# In[31]:


histogram_boxplot(UNR_df['Active Flow Entries'])


# In[32]:


histogram_boxplot(UNR_df['Packets Looked Up'])


# In[33]:


histogram_boxplot(UNR_df['Packets Matched'])


# ### Univariate analysis (discrete numerical)

# In[34]:


# Numerical discrete columns
discrete_num_cols = list(set(num_cols) - set(continuous_num_cols))
discrete_num_cols


# In[35]:


# Mean, Median and Mode
for i in discrete_num_cols:
    mean=UNR_df[i].mean()
    median=UNR_df[i].median()
    mode=UNR_df[i].tolist()[0]
    print(f'Feature: {i}, Mean: {mean}, Median: {median}, Mode: {mode}')


# We will probably drop above features 

# ### Univariate analysis for categorical features

# In[36]:


# Function to create barplots that indicate percentage for each category.
def perc_on_bar(plot, feature):
    '''
    plot
    feature: categorical feature
    the function won't work if a column is passed in hue parameter
    '''
    total = len(feature) # length of the column
    for p in ax.patches:
        percentage = '{:.1f}%'.format(100 * p.get_height()/total) # percentage of each class of the category
        x = p.get_x() + p.get_width() / 2 - 0.05 # width of the plot
        y = p.get_y() + p.get_height()           # hieght of the plot
        ax.annotate(percentage, (x, y), size = 12) # annotate the percantage 
    plt.show() # show the plot


# In[37]:


# 
plt.figure(figsize=(20,5))
ax = sns.countplot(UNR_df['Label'],palette='seismic_r')
perc_on_bar(ax,UNR_df['Label'])


# In[38]:


# 
plt.figure(figsize=(20,5))
ax = sns.countplot(UNR_df['Binary Label'],palette='seismic_r')
perc_on_bar(ax,UNR_df['Binary Label'])


# In[39]:


# 
plt.figure(figsize=(20,5))
ax = sns.countplot(UNR_df['Switch ID'],palette='seismic_r')
perc_on_bar(ax,UNR_df['Switch ID'])


# In[40]:


# 
plt.figure(figsize=(20,5))
ax = sns.countplot(UNR_df['Port Number'],palette='seismic_r')
perc_on_bar(ax,UNR_df['Port Number'])


# ### Bivariate analysis (categorical features)

# In[41]:


## Function to plot stacked bar chart
def stacked_plot(x):
    sns.set(palette='rocket_r')
    tab1 = pd.crosstab(x,UNR_df["Label"],margins=True)
    print(tab1)
    print('-'*120)
    tab2 = pd.crosstab(x,UNR_df["Label"],margins=True, normalize="index")
    print(tab2)
    print('-'*120)
    tab = pd.crosstab(x,UNR_df["Label"],normalize='index')
    tab.plot(kind='bar',stacked=True,figsize=(15,4))
    total = len(x) # length of the column
    plt.legend(loc='lower left', frameon=False)
    plt.legend(loc="upper left", bbox_to_anchor=(1,1))
    plt.xticks(rotation=0)
    plt.show()


# In[42]:


stacked_plot(UNR_df['Port Number'])


# In[43]:


stacked_plot(UNR_df['Binary Label'])


# ### Bivariate Analysis

# In[44]:


plt.figure(figsize=(20,14))

sns.heatmap(UNR_df.corr(),
            annot=True,
            linewidths=.5,
            center=0,
            cbar=False,
            cmap="gist_heat",
            fmt='0.2f')            

plt.show()


# **Comment:** Here we can see plenty of opportunity to drop features that won't impact much on our analysis. Some of them inclusive are filled with only zeros.
# 

# In[45]:


cleanned_df = UNR_df[['Switch ID',
                     'Port Number',
                     'Received Packets',
                     'Received Bytes',
                     'Sent Bytes',
                     'Sent Packets',
                     'Port alive Duration (S)',
                     'Delta Received Packets',
                     'Delta Received Bytes',
                     'Delta Sent Bytes',
                     'Delta Sent Packets',
                     'Delta Port alive Duration (S)',
                     'Connection Point',
                     'Total Load/Rate',
                     'Total Load/Latest',
                     'Unknown Load/Rate',
                     'Unknown Load/Latest',
                     'Latest bytes counter',
                     'Active Flow Entries',
                     'Packets Looked Up',
                     'Packets Matched',
                     'Label',
                     'Binary Label']]


# In[46]:


plt.figure(figsize=(20,14))

sns.heatmap(cleanned_df.corr(),
            annot=True,
            linewidths=.5,
            center=0,
            cbar=False,
            cmap="gist_heat_r",
            fmt='0.2f')            

plt.show()


# In[47]:


# Applying Chi-Square test over target feature Attition_Flag
for i in cat_cols:  
  crosstab = pd.crosstab(UNR_df['Label'],UNR_df[i])  

  Ho = "ProdTaken has --NO-- effect on " + i   # Stating the Null Hypothesis
  Ha = "ProdTaken has an effect on " + i   # Stating the Alternate Hypothesis

  chi, p_value, dof, expected =  stats.chi2_contingency(crosstab)

  if p_value < 0.05:  # Setting our significance level at 5%
      print(f'{Ha} as the p_value ({p_value.round(3)}) < 0.05')
  else:
      print(f'{Ho} as the p_value ({p_value.round(3)}) > 0.05')


# In[48]:


# Saving data prior to Preparation
df_saved = UNR_df.copy()


# In[49]:


# Preprocessing dataFrame
df_start = UNR_df.copy()


# ## Data Pre-processing
# 
# - Prepare data for modeling

# In[50]:


# X has features but not target value
X_prep = df_start.drop('Label', axis=1)

# y only has target value
y_prep = df_start['Label']


# In[51]:


# Transforming categorical features into dummies variables
X_prep_cat = pd.get_dummies(X_prep, drop_first=True)


# In[52]:


le = preprocessing.LabelEncoder()
y_prep = le.fit_transform(y_prep)


# In[53]:


y_prep


# In[54]:


print(X_prep_cat.shape)
print(y_prep.shape)


# In[55]:


# Saving dataset prior to treat for outliers
X_saved = X_prep_cat
y_saved = y_prep


# In[56]:


# First split in Train (0.80) vs. Test (0.20)
x_train,x_test,y_train,y_test=train_test_split(X_prep_cat, y_prep,test_size=0.365)


# ### Model building and evaluation

# In[57]:


logistic_regression=LogisticRegression()
logistic_regression.fit(x_train,y_train)
pred=logistic_regression.predict(x_test)
abc=confusion_matrix(y_test,pred)
logistic_regression=logistic_regression.score(x_test,y_test)
print("The Classification report of logistic regression is : \n",classification_report(y_test,pred))

print('The score of logistic regression is ',logistic_regression*100)


# In[58]:


knn=KNeighborsClassifier()
knn.fit(x_train,y_train)
pred=knn.predict(x_test)
knn_score=knn.score(x_test,y_test)
knn_confusion=confusion_matrix(y_test,pred)
print("The Classification report of KNN is : \n",classification_report(y_test,pred))
print('The KNN classification score is ',knn_score*100)


# In[59]:


d_tree=DecisionTreeClassifier(max_depth=8)
d_tree.fit(x_train,y_train)
pred_dtree=d_tree.predict(x_test)
dtre_pred=d_tree.score(x_test,y_test)
print("The Classification report of Decision Tree is : \n",classification_report(y_test,pred_dtree))
print('The Decision tree score is ',dtre_pred*100)


# In[ ]:




