
# coding: utf-8

# In[8]:


#Loading required packages and libraries for data analysis
import numpy as np
import pandas as pd
import re
import warnings
warnings.filterwarnings('ignore')

#for data visualization
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# In[4]:


#importing the training test datasets
train_df = pd.read_csv('D:/SDGP/Data set/train.csv')
test_df = pd.read_csv('D:/SDGP/Data set/test.csv')


# In[6]:


#look of the training data
train_df.head()


# In[7]:


test_df.head()


# In[9]:


#kind of data we have to work with
train_df.info()


# In[11]:


#printing list of columns in training dataset
train_df.columns


# In[12]:


train_df.describe()


# In[14]:


train_df.describe(include='O')# not zero capital o


# In[16]:


#finding the percentage of missing values in train dataset
train_df.isnull().sum() /len(train_df)*100


# In[17]:


test_df.isnull().sum()/len(test_df)*100


# In[18]:


sns.countplot('Sex',data=train_df)
train_df['Sex'].value_counts()


# In[20]:


#comparing sex feature against survived
#x axis = Independent variable
#y axis = Dependent variable

sns.barplot(x='Sex',y='Survived',data=train_df)
train_df.groupby('Sex',as_index=False).Survived.mean()
#here by using the results can assume that survival rate of female > male


# In[22]:


#comparing pclass(Passenger class) feature against the survived
#passenger classes (1 =1st class, 2= 2nd class, 3= 3rd class)
sns.barplot(x='Pclass',y='Survived',data=train_df)
train_df[["Pclass","Survived"]].groupby(['Pclass'],as_index=False).mean().sort_values(by='Survived',ascending=False)


# In[23]:


#Comparing the Embarked feature against Survived
sns.barplot(x='Embarked',y='Survived',data=train_df)
train_df[["Embarked", "Survived"]].groupby(['Embarked'], as_index=False).mean().sort_values(by='Survived', ascending=False)


# In[24]:


#Effects of having parenets or childeren on board
sns.barplot(x='Parch',y='Survived',data=train_df)
train_df[["Parch", "Survived"]].groupby(['Parch'], as_index=False).mean().sort_values(by='Survived', ascending=False)


# In[25]:


#Effect of having spouse or sibling on survival
sns.barplot(x='SibSp',y='Survived',data=train_df)
train_df[["SibSp", "Survived"]].groupby(['SibSp'], as_index=False).mean().sort_values(by='Survived', ascending=False)


# In[26]:


#Age column has some missing values.
#we will care of that later when we clean our training data
train_df.Age.hist(bins=10,color='teal')
plt.xlabel('Age')
plt.ylabel('Count')
plt.show
print("The Median age of passengers is :",int(train_df.Age.median()))
print("The standard Deviation age of passengers is :",int(train_df.Age.std()))


# In[27]:


sns.lmplot(x='Age',y='Survived',data=train_df,palette='Set1')
#this clearly shows whether younger or edults are more likely to survive


# In[28]:


sns.lmplot(x='Age',y='Survived',data=train_df,hue='Sex',palette='Set1')
#this shows the male and female suvival with their ages too


# In[29]:


#Checking for outliers in Age data
sns.boxplot(x='Sex',y='Age',data=train_df)

#getting the median age according to Sex
train_df.groupby('Sex',as_index=False)['Age'].median()


# In[30]:


#plotting the Fare column to see the spread of data
sns.boxplot("Fare",data=train_df)

#Checking the mean and median values
print("Mean value of Fare is :",train_df.Fare.mean())
print("Median value of Fare is :",train_df.Fare.median())


# In[ ]:


#cleaning data

