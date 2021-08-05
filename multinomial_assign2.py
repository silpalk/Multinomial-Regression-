# -*- coding: utf-8 -*-

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
import numpy as np

mode = pd.read_csv(r"C:\Users\asilp\Desktop\datascience\assign11\loan.csv")
mode.head(10)
mode.columns
mode.dtypes
mode.describe()
mode1=mode.dropna(axis=1,how='all',inplace=False)
mode1=mode.select_dtypes(np.number)
mode1.isnull().sum()
mode1.isna

mode1.drop(['tot_hi_cred_lim','total_bal_ex_mort','total_bc_limit','total_il_high_credit_limit'],axis=1,inplace=True)
from sklearn.impute import SimpleImputer
mean_imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
mode1["collections_12_mths_ex_med"] = pd.DataFrame(mean_imputer.fit_transform(mode1[["collections_12_mths_ex_med"]]))
mode1["chargeoff_within_12_mths"] = pd.DataFrame(mean_imputer.fit_transform(mode1[["chargeoff_within_12_mths"]]))
mode1["pub_rec_bankruptcies"] = pd.DataFrame(mean_imputer.fit_transform(mode1[["pub_rec_bankruptcies"]]))
mode1["tax_liens"] = pd.DataFrame(mean_imputer.fit_transform(mode1[["tax_liens"]]))
mode1["mths_since_last_delinq"] = pd.DataFrame(mean_imputer.fit_transform(mode1[["mths_since_last_delinq"]]))
mode1["mths_since_last_record"] = pd.DataFrame(mean_imputer.fit_transform(mode1[["mths_since_last_record"]]))


mode1.isnull().sum()

#cheching for categorical data
mode2=mode.select_dtypes('O')
mode2.columns
mode2['application_type'].value_counts()
mode2['initial_list_status'].value_counts()
mode2.drop(['application_type','term','emp_length','zip_code','url','issue_d'],axis=1,inplace=True)
mode2.drop(['emp_title'],axis=1,inplace=True)
mode2.drop(['sub_grade','desc','title','int_rate'],axis=1,inplace=True)
mode2=mode2.iloc[:,:6]
mode2['initial_list_status'].value_counts()
mode2['pymnt_plan'].value_counts()
mode2.drop(['pymnt_plan'],axis=1,inplace=True)


from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
mode2.columns
mode2['grade']=le.fit_transform(mode['grade'])
mode2['home_ownership']=le.fit_transform(mode2['home_ownership'])
mode2['verification_status']=le.fit_transform(mode2['verification_status'])
mode2['loan_status']=le.fit_transform(mode2['loan_status'])
mode2['purpose']=le.fit_transform(mode2['purpose'])

data_final=pd.concat([mode1,mode2],axis=1)

data_final.isnull().sum()
# Boxplot of independent variable distribution for each category of choice 
sns.boxplot(x = "loan_status", y = "total_pymnt", data = data_final)
sns.boxplot(x = "loan_status", y = "revol_bal", data = data_final)
sns.boxplot(x = "loan_status", y = "funded_amnt", data = data_final)
sns.boxplot(x = "loan_status", y = "installment", data = data_final)
sns.boxplot(x = "loan_status", y = "inq_last_6mths", data = data_final)




# Scatter plot for each categorical choice of car
sns.stripplot(x = "loan_status", y = "total_pymnt",jitter=True, data = data_final)
sns.scatterplot((x = "loan_status", y = "revol_bal",jitter=True, data = data_final)
sns.scatterplot(x = "loan_status", y = "funded_amnt",jitter=True, data = data_final)
sns.scatterplot(x = "loan_status", y = "installment",jitter=True ,data = data_final)
sns.stripplot(x = "loan_status", y = "inq_last_6mths",jitter=True, data = data_final)


# Scatter plot between each possible pair of independent variable and also histogram for each independent variable 
sns.pairplot(data_final) # Normal
sns.pairplot(data_final, hue = "loan_status") # With showing the category of each car choice in the scatter plot

# Correlation values between each independent features
mode['emp_title'].dtypes
data_final.columns
mode.corr()
x=data_final.drop(['loan_status'],axis='columns')
y=data_final.iloc[:,[36]]
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
