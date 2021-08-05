# -*- coding: utf-8 -*-
"""
Created on Fri Mar 19 22:22:09 2021

@author: asilp
"""

### Multinomial Regression ####
import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

mode = pd.read_csv(r"C:\Users\asilp\Desktop\datascience\assign11\mdata.csv")
mode.head(10)
mode.columns
mode.dtypes
mode.describe()
mode.prog.value_counts()
mode.drop(['Unnamed: 0'],axis=1,inplace=True)
from sklearn.preprocessing import LabelEncoder,OneHotEncoder
le=LabelEncoder()
enc=OneHotEncoder(drop='if_binary')
mode['ses']=le.fit_transform(mode['ses'])
mode['female']=pd.get_dummies(mode['female'])
mode['honors']=pd.get_dummies(mode['honors'])
mode['schtyp']=le.fit_transform(mode['schtyp'])
mode['prog']=le.fit_transform(mode['prog'])
# Boxplot of independent variable distribution for each category of choice 
sns.boxplot(x = "prog", y = "honors", data = mode)
sns.boxplot(x = "prog", y = "read", data = mode)
sns.boxplot(x = "prog", y = "write", data = mode)
sns.boxplot(x = "prog", y = "math", data = mode)
sns.boxplot(x = "prog", y = "schtyp", data = mode)
sns.boxplot(x = "prog", y = "science", data = mode)



# Scatter plot for each categorical choice of car
sns.stripplot(x = "prog", y = "math", jitter = True, data = mode)
sns.stripplot(x = "prog", y = "schtyp", jitter = True, data = mode)
sns.stripplot(x = "prog", y = "science", jitter = True, data = mode)
sns.stripplot(x = "prog", y = "read", jitter = True, data = mode)
sns.stripplot(x = "prog", y = "write", jitter = True, data = mode)
sns.stripplot(x = "prog", y = "honors", jitter = True, data = mode)


# Scatter plot between each possible pair of independent variable and also histogram for each independent variable 
sns.pairplot(mode) # Normal
sns.pairplot(mode, hue = "prog") # With showing the category of each car choice in the scatter plot

# Correlation values between each independent features
mode.corr()
x=mode.drop(['prog'],axis='columns')
y=mode.iloc[:,[4]]
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size = 0.2)

# ‘multinomial’ option is supported only by the ‘lbfgs’ and ‘newton-cg’ solvers

model = LogisticRegression(multi_class = "multinomial", solver = "newton-cg")
model.fit(x_train,y_train)

y_test_predict = model.predict(x_test) # Test predictions

# Test accuracy 
accuracy_score(y_test_predict,y_test)

y_train_predict = model.predict(x_train) # Train predictions 
# Train accuracy 
accuracy_score(y_train_predict,y_train) 
