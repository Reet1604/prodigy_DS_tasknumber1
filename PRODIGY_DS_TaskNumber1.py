#!/usr/bin/env python
# coding: utf-8

# In[ ]:


Author - Reetu Khatri

Overview
In this Titanic dataset
The data has been split into two groups:

training set (train.csv)
test set (test.csv)
The training set should be used to build your machine learning models. For the training set, we provide the outcome (also known as the “ground truth”) for each passenger. Your model will be based on “features” like passengers’ gender and class. You can also use feature engineering to create new features.

The test set should be used to see how well your model performs on unseen data. For the test set, we do not provide the ground truth for each passenger. It is your job to predict these outcomes. For each passenger in the test set, use the model you trained to predict whether or not they survived the sinking of the Titanic.

We also include gender_submission.csv, a set of predictions that assume all and only female passengers survive, as an example of what a submission file should look likeOverview
1) Understand the shape of the data (Histograms, box plots, etc.)
2) Data Cleaning
3) Data Exploration
# # Importing library

# In[1]:


import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


# # Loading  the data

# In[5]:


train_data= pd.read_csv('train.csv')
train_data


# In[7]:


test_data=pd.read_csv('test.csv')
test_data


# In[14]:


gender_data=pd.read_csv('gender_submission.csv')
gender_data


# In[19]:


train_data['train_test']=1
test_data['train_test']=0
test_data['Survived']=np.NaN
all_data=pd.concat([train_data,test_data])

get_ipython().run_line_magic('matplotlib', 'inline')
all_data.columns

Concatenate Two Columns Using + Operator in pandas

By use + operator simply you can concatenate two or multiple text/string columns in pandas DataFrame. Note that when you apply + operator on numeric columns it actually does addition instead of concatenation.
# # Project Planning

When starting any project, I like to outline the steps that I plan to take. Below is the rough outline that I created for this project using commented cells.
# In[ ]:


# Understand nature of the data .info() .describe()
# Histograms and boxplots 
# Value counts 
# Missing data 
# Correlation between the metrics 
# Explore interesting themes 
    # Wealthy survive? 
    # By location 
    # Age scatterplot with ticket price 
    # Young and wealthy Variable? 
    # Total spent? 
# Feature engineering 
# preprocess data together or use a transformer? 
    # use label for train and test   
# Scaling?

# Model Baseline 
# Model comparison with CV

Light Data Exploration
1) For numeric data
Made histograms to understand distributions
Corrplot
Pivot table comparing survival rate across numeric variables
2) For Categorical Data
Made bar charts to understand balance of classes
Made pivot tables to understand relationship with survivalSingle Variable Plots
Histograms
Box Plots
Bar charts
Relationships & Multi-variable plots
Scatterplots
Correlation Matrices
Pivot Tables
Bar Charts
Line Charts
# # EDA

# In[20]:


# look at our data types & null counts 
train_data.info()


# In[21]:


test_data.info()


# In[22]:


# to better understand the numeric data, we want to use the .describe() method. This gives us an understanding of the central tendencies of the data 
train_data.describe()


# In[25]:


#quick way to separate numeric columns
train_data.describe().columns


# In[26]:


train_data.head()


# In[28]:


# look at numeric and categorical values separately 
df_num = train_data[['Age','SibSp','Parch','Fare']]
df_cat = train_data[['Survived','Pclass','Sex','Ticket','Cabin','Embarked']]


# In[29]:


df_num.head(3)


# In[30]:


df_cat.head(3)


# In[36]:


# distribution for AGE by histogram
plt.hist(train_data['Age'])
plt.title('Age')
plt.show()


# In[31]:


#distributions for all numeric variables 
for i in df_num.columns:
    plt.hist(df_num[i])
    plt.title(i)
    plt.show()


# In[ ]:


# finding shape of both  dataset


# In[11]:


train_data.shape  # here 891 rows and 12 columns in this dataset


# In[12]:


test_data.shape  # here 418 rows and 11 columns in dataset


# In[15]:


gender_data.shape  # here 418 rows and 2columns in this dataset


# In[ ]:


# Finding null values


# In[9]:


train_data.isnull().sum()   # here in age and cabin have null values


# In[16]:


test_data.isnull().sum()   # here also null values in age and cabin columns


# In[17]:


gender_data.isnull().sum()


# In[37]:


print(df_num.corr())
sns.heatmap(df_num.corr())


# In[39]:


# compare survival rate across Age, SibSp, Parch, and Fare 
pd.pivot_table(train_data, index = 'Survived', values = ['Age','SibSp','Parch','Fare'])


Remember that the sample submission file in gender_submission.csv assumes that all female passengers survived (and all male passengers died).

Is this a reasonable first guess? We'll check if this pattern holds true in the data (in train.csv).

Copy the code below into a new code cell. Then, run the cell.
# In[46]:


df_cat.head(3)


# In[49]:


df_cat.value_counts()


# In[44]:


for i in df_cat.columns:
    sns.barplot(df_cat[i].value_counts().index,df_cat[i].value_counts()).set_title(i)
    plt.show()

Cabin and ticket graphs are very messy. This is an area where we may want to do some feature engineering!
# In[52]:


# Comparing survival and each of these categorical variables 
print(pd.pivot_table(train_data,index='Survived',columns= 'Pclass', values= 'Ticket',aggfunc='count'))


# In[53]:


print(pd.pivot_table(train_data,index='Survived',columns='Sex',values ='Ticket',aggfunc ='count'))


# In[54]:


print(pd.pivot_table(train_data,index="Survived" , columns= 'Embarked', values ='Ticket',aggfunc='count'))


# # Feature Engineering

# In[58]:


df_cat.head(2)


# In[59]:


train_data.head(2)


1) Cabin - Simplify cabins (evaluated if cabin letter (cabin_adv) or the purchase of tickets across multiple cabins (cabin_multiple) impacted survival)
2) Tickets - Do different ticket types impact survival rates?
3) Does a person's title relate to survival rates?
# In[57]:


# after looking at this, we may want to look at cabin by letter or by number. Let's create some categories for this 
# letters 
# multiple letters 

df_cat.Cabin
train_data['cabin_multiple'] = train_data.Cabin.apply(lambda x: 0 if pd.isna(x) else len(x.split(' ')))


train_data['cabin_multiple'].value_counts()


# In[60]:


pd.pivot_table(train_data, index = 'Survived', columns = 'cabin_multiple', values = 'Ticket' ,aggfunc ='count')


# In[67]:


#create plot with matplotlib
plt.scatter(x=train_data['Age'],y= train_data['Survived'])
plt.title('survived according the age')
plt.xlabel('Age')
plt.ylabel('Survived')
plt.legend('best')
plt.show()


# In[77]:


train_data['Sex']


# In[83]:


#create plot with matplotlib
plt.scatter(x=train_data['Sex'],y= train_data['Age'],color="red")
plt.title('sex v/s age')
plt.xlabel('Sex')
plt.ylabel('Age')
plt.legend('best')
plt.show()


# In[84]:


#creates categories based on the cabin letter (n stands for null)
#in this case we will treat null values like it's own category

train_data['cabin_adv'] = train_data.Cabin.apply(lambda x: str(x)[0])


# In[85]:


train_data['cabin_adv']


# In[87]:


#understand ticket values better 
#numeric vs non numeric 
train_data['numeric_ticket'] = train_data.Ticket.apply(lambda x: 1 if x.isnumeric() else 0)
train_data['ticket_letters'] = train_data.Ticket.apply(lambda x: ''.join(x.split(' ')[:-1]).replace('.','').replace('/','').lower() if len(x.split(' ')[:-1]) >0 else 0)


# In[88]:


train_data['numeric_ticket'] 


# In[89]:


train_data['ticket_letters']


# In[90]:


train_data['numeric_ticket'].value_counts()


# In[92]:


#lets us view all rows in dataframe through scrolling. This is for convenience 
pd.set_option("max_rows", None)
train_data['ticket_letters'].value_counts()


# In[93]:


#difference in numeric vs non-numeric tickets in survival rate 
pd.pivot_table(train_data,index='Survived',columns='numeric_ticket', values = 'Ticket', aggfunc='count')


# In[95]:


#survival rate across different tyicket types 
pd.pivot_table(train_data,index='Survived',columns='ticket_letters', values = 'Ticket', aggfunc='count')


# In[96]:


#feature engineering on person's title 
train_data.Name.head(50)
train_data['name_title'] = train_data.Name.apply(lambda x: x.split(',')[1].split('.')[0].strip())
#mr., ms., master. etc


# In[97]:


train_data['name_title'].value_counts()


# # Data Preprocessing for Model
Data Preprocessing for Model
1) Drop null values from Embarked (only 2)

2) Include only relevant variables (Since we have limited data, I wanted to exclude things like name and passanger ID so that we could have a reasonable number of features for our models to deal with)
Variables: 'Pclass', 'Sex','Age', 'SibSp', 'Parch', 'Fare', 'Embarked', 'cabin_adv', 'cabin_multiple', 'numeric_ticket', 'name_title'

3) Do categorical transforms on all data. Usually we would use a transformer, but with this approach we can ensure that our traning and test data have the same colums. We also may be able to infer something about the shape of the test data through this method. I will stress, this is generally not recommend outside of a competition (use onehot encoder).

4) Impute data with mean for fare and age (Should also experiment with median)

5) Normalized fare using logarithm to give more semblance of a normal distribution

6) Scaled data 0-1 with standard scaler
# In[98]:


#create all categorical variables that we did above for both training and test sets 
all_data['cabin_multiple'] = all_data.Cabin.apply(lambda x: 0 if pd.isna(x) else len(x.split(' ')))
all_data['cabin_adv'] = all_data.Cabin.apply(lambda x: str(x)[0])
all_data['numeric_ticket'] = all_data.Ticket.apply(lambda x: 1 if x.isnumeric() else 0)
all_data['ticket_letters'] = all_data.Ticket.apply(lambda x: ''.join(x.split(' ')[:-1]).replace('.','').replace('/','').lower() if len(x.split(' ')[:-1]) >0 else 0)
all_data['name_title'] = all_data.Name.apply(lambda x: x.split(',')[1].split('.')[0].strip())

#impute nulls for continuous data 
#all_data.Age = all_data.Age.fillna(training.Age.mean())
all_data.Age = all_data.Age.fillna(train_data.Age.median())
#all_data.Fare = all_data.Fare.fillna(training.Fare.mean())
all_data.Fare = all_data.Fare.fillna(train_data.Fare.median())

#drop null 'embarked' rows. Only 2 instances of this in training and 0 in test 
all_data.dropna(subset=['Embarked'],inplace = True)

#tried log norm of sibsp (not used)
all_data['norm_sibsp'] = np.log(all_data.SibSp+1)
all_data['norm_sibsp'].hist()

# log norm of fare (used)
all_data['norm_fare'] = np.log(all_data.Fare+1)
all_data['norm_fare'].hist()

# converted fare to category for pd.get_dummies()
all_data.Pclass = all_data.Pclass.astype(str)

#created dummy variables from categories (also can use OneHotEncoder)
all_dummies = pd.get_dummies(all_data[['Pclass','Sex','Age','SibSp','Parch','norm_fare','Embarked','cabin_adv','cabin_multiple','numeric_ticket','name_title','train_test']])

#Split to train test again
X_train = all_dummies[all_dummies.train_test == 1].drop(['train_test'], axis =1)
X_test = all_dummies[all_dummies.train_test == 0].drop(['train_test'], axis =1)


y_train = all_data[all_data.train_test==1].Survived
y_train.shape


# In[99]:


# Scale data 
from sklearn.preprocessing import StandardScaler
scale = StandardScaler()
all_dummies_scaled = all_dummies.copy()
all_dummies_scaled[['Age','SibSp','Parch','norm_fare']]= scale.fit_transform(all_dummies_scaled[['Age','SibSp','Parch','norm_fare']])
all_dummies_scaled

X_train_scaled = all_dummies_scaled[all_dummies_scaled.train_test == 1].drop(['train_test'], axis =1)
X_test_scaled = all_dummies_scaled[all_dummies_scaled.train_test == 0].drop(['train_test'], axis =1)

y_train = all_data[all_data.train_test==1].Survived


# In[ ]:


# Model Building (Baseline Validation Performance)


# In[100]:


from sklearn.model_selection import cross_val_score
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn import tree


# In[101]:


#  usually use Naive Bayes as a baseline   classification tasks 
gnb = GaussianNB()
cv = cross_val_score(gnb,X_train_scaled,y_train,cv=5)
print(cv)
print(cv.mean())


# In[103]:


lr = LogisticRegression(max_iter = 2000)
cv = cross_val_score(lr,X_train,y_train,cv=5)
print(cv)
print(cv.mean())


# In[104]:


lr = LogisticRegression(max_iter = 2000)
cv = cross_val_score(lr,X_train_scaled,y_train,cv=5)
print(cv)
print(cv.mean())


# In[105]:


dt = tree.DecisionTreeClassifier(random_state = 1)
cv = cross_val_score(dt,X_train,y_train,cv=5)
print(cv)
print(cv.mean())


# In[ ]:




