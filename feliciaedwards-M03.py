#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 14 2019

@author: feliciapedwards
"""


# Narrative:
# In this dataset, there are 32561 observations and 15 attributes in the original set of data
# 9 of the 15 attributes are objects, the remaining 6 are integers
# The source of this data is from the following website: http://mlr.cs.umass.edu 
# I have removed obsolte columns, they provide either redundant data or no insights
# I decided to plot workclass, education, and occupation as those provided the most interesting information. 
# I one-hot encoded, sex, marital status and income. 
# I normalized capital-gain and capital-loss, so we could compare the two.
# I binned the age column to give more perspective and insights.
# I replaced unknowns or question marks into various categories, usually "other"

# Binary Question: Based on the features provided, is the person male or female
# The 'sex' column will be the expert label

import csv
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import seaborn as sns
from pandas.tools.plotting import scatter_matrix
from matplotlib import cm
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
# train a logistic regression model on the training set
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
from sklearn.ensemble import RandomForestClassifier as RFC



#Dataset
url = "http://mlr.cs.umass.edu/ml/machine-learning-databases/adult/adult.data"
adult = pd.read_csv(url, header=None)
adult.columns = ["age", "workclass", "fnlwgt", "education", "education-num", "ms", "occupation", "relationship", "race", "sex", "cg", "cl", "hours-per-week", "native-country", "Income"]

# Check the first rows of the data frame
adult.head()

# Check the number of rows and columns
adult.shape

#check types
adult.dtypes

# Replace >Question Marks< with NaNs
adult = adult.replace(to_replace=" ?", value=float("NaN"))

# Count NaNs
adult.isnull().sum()



# Check unique values of race, doesn't look like it needs to be consolidated to me 
adult.loc[:,"race"].unique()


# Check unique values of hours
adult.loc[:,"hours-per-week"].unique()

l = adult.loc[:,"hours-per-week"].unique()
l.sort()
l




# One-hot encode categorical data for income
# Check the unique values
adult.loc[:, "Income"].unique()

# Create 2 new columns, one for each state in Income
# Assumption: there are only two options for income, either above 50k or below 50k
adult.loc[:, " <=50K"] = (adult.loc[:, "Income"] == " <=50K").astype(int)
adult.loc[:, " >50K"] = (adult.loc[:, "Income"] == " >50K").astype(int)

# Get the counts for each value
adult.loc[:,"Income"].value_counts()

# Check the number of rows and columns
adult.shape

# Check the first rows of the data frame
adult.head()

# Remove obsolete column
adult = adult.drop("Income", axis=1)

# Check the number of rows and columns to make sure "Income" column was removed
adult.shape
#######################



# One-hot encode categorical data for marital status
# Check the unique values
# Assumption about marital status, you either have a significant other regardless of 
# being legally married. 
adult.loc[:, "ms"].unique()

# One-hot encode categorical data with 3 types
Replace = adult.loc[:, "ms"] == " Never-married"
adult.loc[Replace, "ms"] = "1"

Replace = adult.loc[:, "ms"] == " Married-civ-spouse"
adult.loc[Replace, "ms"] = "2"

Replace = adult.loc[:, "ms"] == " Divorced"
adult.loc[Replace, "ms"] = "1"

Replace = adult.loc[:, "ms"] == " Married-AF-spouse"
adult.loc[Replace, "ms"] = "2"

Replace = adult.loc[:, "ms"] == " Married-spouse-absent"
adult.loc[Replace, "ms"] = "2"

Replace = adult.loc[:, "ms"] == " Separated"
adult.loc[Replace, "ms"] = "1"

Replace = adult.loc[:, "ms"] == " Widowed"
adult.loc[Replace, "ms"] = "1"

# Check the unique values
adult.loc[:, "ms"].unique()

# Plot the counts for each category
adult.loc[:,"ms"].value_counts().plot(kind='bar')





# Check unique occupations available
adult.loc[:, "occupation"].unique()

# Plot the counts for each category
adult.loc[:,"occupation"].value_counts().plot(kind='bar')



# Check unique workclass available
adult.loc[:, "workclass"].unique()

# Plot the counts for each category
adult.loc[:,"workclass"].value_counts().plot(kind='bar')



# Check the unique values
adult.loc[:, "education"].unique()

# Get the counts for each value
adult.loc[:, "education"].value_counts()

# Plot the counts for each category
adult.loc[:,"education"].value_counts().plot(kind='bar')





# One-hot encode categorical data for sex
# Check the unique values
# Assumption: I'm assuming the sex is decided by the individual, which pronoun the individual associates with,
# not necessarily the sex the individual was born with or is currently transitioning from.
adult.loc[:, "sex"].unique()

# Replace female with 1
Replace = adult.loc[:, "sex"] == " Female"
adult.loc[Replace, "sex"] = " 1"

# Replace male with 0
Replace = adult.loc[:, "sex"] == " Male"
adult.loc[Replace, "sex"] = " 0"

# Get the counts for each value
adult.loc[:,"sex"].value_counts()

# Check the number of rows and columns
adult.shape

# Check the first rows of the data frame
adult.head()
#######################




# Remove obsolete columns
# Some of this data is redundant
# Some of this data isn't informative or doesn't lead to insights
adult = adult.drop("fnlwgt", axis=1)
adult = adult.drop("education-num", axis=1)
adult = adult.drop("relationship", axis=1)
adult = adult.drop("hours-per-week", axis=1)
adult = adult.drop("native-country", axis=1)




# Check the first rows of the data frame
adult.head()

# Check the number of rows and columns
adult.shape

#check types
adult.dtypes

adult['ms'] = adult.ms.astype(int)
adult['sex'] = adult.sex.astype(int)
adult['Income'] = adult.Income.astype(int)
adult['education'] = adult.education.astype(int)
adult.dtypes

#Create copy of dataframe adult
adult_tr = adult


columns = ['age', 'ms', 'cg', 'cl', ' >50K', ' <=50K']

# Defining X and y 
X = stats.zscore(adult_tr[columns])
print((adult_tr[columns]))
y = adult_tr.sex
print(y)

#Cluster the data
#Here I choose to have 2 clusters
kmeans = KMeans(n_clusters=2, random_state=0).fit(X)
labels = kmeans.labels_

#Include cluster labels in original data
adult_tr['clusters'] = labels

#Add the column 
columns.extend(['clusters'])

#Analyze the clusters
print(adult_tr[columns].groupby(['clusters']).mean())


# Summary Comment block on cluster labels:
# Here we can see means of each cluster based on the features I chose above: age, capital-gain, marital status, education and income.
# In the first cluster, the average age is roughly 37, the capital gain is almost 149, the capital loss is 53, and most of these
# more individuals are single, and earn less than 50k/year. The second cluster has an average age of 44, 
# capital gain of 4006, capital loss of 195 and mostly include individuals who have a spouse, have an income greater than 50k.

X0 = X
y0 = y

#Create Training Testing and Scaling
X0_train, X0_test, y0_train, y0_test = train_test_split(X0, y0, train_size=0.75, test_size=0.25, random_state=0, shuffle=True)

# instantiate model
logreg = LogisticRegression()

# fit model
logreg.fit(X0_train, y0_train)

# make class predictions for the testing set
# Classification accuracy: percentage of correct predictions
y0_pred_class = logreg.predict(X0_test)

# calculate accuracy
print(metrics.accuracy_score(y0_test, y0_pred_class))
# 71% accuracy

# examine the class distribution of the testing set (using a Pandas Series method)
y0_test.value_counts()

# calculate the percentage of ones (females)
# because y_test only contains ones and zeros, we can simply calculate the mean = percentage of ones
y0_test.mean()

# calculate the percentage of zeros (males)
1 - y0_test.mean()



# calculate null accuracy in a single line of code
# only for binary classification problems coded as 0/1
# This means that a dumb model that always predicts 0 would be right 67% of the time
# This shows how classification accuracy is not that good as it's close to a dumb model
# It's a good way to know the minimum we should achieve with our models
max(y0_test.mean(), 1 - y0_test.mean())

# calculate null accuracy (for multi-class classification problems)
# Null accuracy: accuracy that could be achieved by always predicting the most frequent class 
y0_test.value_counts().head(1) / len(y0_test)

# IMPORTANT: first argument is true values, second argument is predicted values
# this produces a 2x2 numpy array (matrix)
print(metrics.confusion_matrix(y0_test, y0_pred_class))

# print the first 25 true and predicted responses
print('True', y0_test.values[0:25])
print('Pred', y0_pred_class[0:25])

# save confusion matrix and slice into four pieces
confusion = metrics.confusion_matrix(y0_test, y0_pred_class)
print(confusion)


#[row, column]
TP = confusion[1, 1]
TN = confusion[0, 0]
FP = confusion[0, 1]
FN = confusion[1, 0]


# use float to perform true division, not integer division
# Classification Accuracy: Overall, how often is the classifier correct?
print((TP + TN) / float(TP + TN + FP + FN))

#The model is correct roughly 71% of the time, which is better than the null-accuracy of 67%

# Classification Error: Overall, how often is the classifier incorrect?
classification_error = (FP + FN) / float(TP + TN + FP + FN)
print(classification_error)

# Sensitivity/Recall: When the actual value is positive, how often is the prediction correct?
# Recall is usually used when the goal is to limit the number of false negatives 
sensitivity = TP / float(FN + TP)
print(sensitivity)

#Specificity: When the actual value is negative, how often is the prediction correct?
specificity = TN / (TN + FP)
print(specificity)

# False Positive Rate: When the actual value is negative, how often is the prediction incorrect?
false_positive_rate = FP / float(TN + FP)
print(false_positive_rate)

# Precision: When a positive value is predicted, how often is the prediction correct?
precision = TP / float(TP + FP)
print(precision)

# F1-Score = 2 * (precision*recall/(precision+recall))
# It is useful when you need to take both precision and recall into account
f1 = (2* (TP / float(TP + FP)) * (TP / float(FN + TP)))/((TP / float(TP + FP)) + (TP / float(FN + TP)))
print(f1)


# print the first 10 predicted responses
# 1D array (vector) of binary values (0, 1)
logreg.predict(X0_test)[0:10]

# print the first 10 predicted probabilities of class membership
logreg.predict_proba(X0_test)[0:10]

# print the first 10 predicted probabilities for class 1
logreg.predict_proba(X0_test)[0:10, 1]

# store the predicted probabilities for class 1
y0_pred_prob = logreg.predict_proba(X0_test)[:, 1]

# store the predicted probabilities for both classes
y0_pred_prob_both = logreg.predict_proba(X0_test)[:,:]
print(y0_pred_prob_both)


# This classification model predicts if the sex is male or female based on income, marital status, capital gain, capital loss, and age. 
# The null accuracy for this model is 67%, which means that a dumb model that always predicts 0 would be right 67% of the time. This 
# model is correct roughly 71% of the time, so its 4% more accurate than a dumb model, so it does add some value. 


# first argument is true values, second argument is predicted probabilities

# we use y_test and y_pred_prob
# we don't use y_pred_class, because it will yield incorrect results without generating an error
# roc_curve returns 3 objects: fpr, tpr, threshold
# fpr: false positive rate
# tpr: true positive rate


# AUC is the percentage of the ROC plot that is underneath the curve:
# Higher value = better 
# Useful even when there is high class imbalance
auc = roc_auc_score(y0_test, y0_pred_prob)
print('AUC: %.2f' % auc)
# AUC is 75% 

# Receiver Operating Characteristic (ROC) curve can help you to choose a threshold that balances sensitivity and specificity
# Q: How are sensitivity and specificity affected by various thresholds
fpr, tpr, thresholds = roc_curve(y0_test, y0_pred_prob)
plt.plot([0, 1], [0, 1], linestyle='--') #no skill
plt.plot(fpr, tpr, label='ROC Curve with AUC area of %0.2f' % auc)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.0])
plt.rcParams['font.size'] = 12
plt.title('ROC curve for sex classifier')
plt.xlabel('False Positive Rate (1 - Specificity)')
plt.ylabel('True Positive Rate (Sensitivity)')
plt.legend(loc="lower right")
plt.grid(True)

# Accuracy Summary
# This model has an AUC of .75, and is thus accurate 75% of the time.
# The model is incorrect 29% of the time. 
# When the actual value is positive, the prediction is correct 58% of the time.
# When the actual value is negative, the prediction is correct 77% of the time. 
# When the actual value is negative, the prediction is incorrect 23% of the time. 
# When a positive value is predicted, the prediction is correct 56% of the time. 
# F1 Score of 57%, which takes both precision and sensitivity into account.

# I would choose a threshold of 0.4 as there isn't a huge difference from 0.4 to 0.3. a threshold of 0.4 
# yields a sensitivity of 0.82 and specificity of 0.62. It's a good balance between both sensitivity and specificity


###########################################
#Random Forest

X1 = X
y1 = y

#Create Training Testing and Scaling
X1_train, X1_test, y1_train, y1_test = train_test_split(X1, y1, random_state=0)

# instantiate model
rfc= RFC()

# fit model
rfc.fit(X1_train, y1_train)

# make class predictions for the testing set
# Classification accuracy: percentage of correct predictions
y1_pred_class = rfc.predict(X1_train)

# calculate accuracy
# train accuracy
print(metrics.accuracy_score(y1_train, y1_pred_class))
# 74% accuracy - more accurate than the logistic regression classifier 

# examine the class distribution of the testing set (using a Pandas Series method)
y1_test.value_counts()

# calculate the percentage of ones (females)
# because y_test only contains ones and zeros, we can simply calculate the mean = percentage of ones
y1_test.mean()

# calculate the percentage of zeros (males)
1 - y1_test.mean()



# calculate null accuracy in a single line of code
# only for binary classification problems coded as 0/1
# This means that a dumb model that always predicts 0 would be right 67% of the time
# This shows how classification accuracy is not that good as it's close to a dumb model
# It's a good way to know the minimum we should achieve with our models
max(y1_test.mean(), 1 - y1_test.mean())

# calculate null accuracy (for multi-class classification problems)
# Null accuracy: accuracy that could be achieved by always predicting the most frequent class 
y1_test.value_counts().head(1) / len(y1_test)

# IMPORTANT: first argument is true values, second argument is predicted values
# this produces a 2x2 numpy array (matrix)
print(metrics.confusion_matrix(y1_train, y1_pred_class))

# print the first 25 true and predicted responses
print('True', y1_test.values[0:25])
print('Pred', y1_pred_class[0:25])


# save confusion matrix and slice into four pieces
confusion1 = metrics.confusion_matrix(y1_train, y1_pred_class)
print(confusion1)


#[row, column]
TP1 = confusion1[1, 1]
TN1 = confusion1[0, 0]
FP1 = confusion1[0, 1]
FN1 = confusion1[1, 0]


# use float to perform true division, not integer division
# Classification Accuracy: Overall, how often is the classifier correct?
print((TP1 + TN1) / float(TP1 + TN1 + FP1 + FN1))

#The model is correct roughly 74% of the time, which is better than the null-accuracy of 67%

# Classification Error: Overall, how often is the classifier incorrect?
classification_error = (FP1 + FN1) / float(TP1 + TN1 + FP1 + FN1)
print(classification_error)

# Sensitivity/Recall: When the actual value is positive, how often is the prediction correct?
# Recall is usually used when the goal is to limit the number of false negatives 
sensitivity = TP1 / float(FN1 + TP1)
print(sensitivity)

#Specificity: When the actual value is negative, how often is the prediction correct?
specificity = TN1 / (TN1 + FP1)
print(specificity)

# False Positive Rate: When the actual value is negative, how often is the prediction incorrect?
false_positive_rate = FP1 / float(TN1 + FP1)
print(false_positive_rate)

# Precision: When a positive value is predicted, how often is the prediction correct?
precision = TP1 / float(TP1 + FP1)
print(precision)

# F1-Score = 2 * (precision*recall/(precision+recall))
# It is useful when you need to take both precision and recall into account
f1_rfc = (2* (TP1 / float(TP1 + FP1)) * (TP1 / float(FN1 + TP1)))/((TP1 / float(TP1 + FP1)) + (TP1 / float(FN1 + TP1)))
print(f1_rfc)


# print the first 10 predicted responses
# 1D array (vector) of binary values (0, 1)
rfc.predict(X1_test)[0:10]

# print the first 10 predicted probabilities of class membership
rfc.predict_proba(X1_test)[0:10]

# print the first 10 predicted probabilities for class 1
rfc.predict_proba(X1_test)[0:10, 1]

# store the predicted probabilities for class 1
y1_pred_prob = rfc.predict_proba(X1_test)[:, 1]

# store the predicted probabilities for both classes
y1_pred_prob_both = rfc.predict_proba(X1_test)[:,:]
print(y1_pred_prob_both)


# This classification model predicts if the sex is male or female based on income, marital status, capital gain, capital loss, and age. 
# The null accuracy for this model is 67%, which means that a dumb model that always predicts 0 would be right 67% of the time. This 
# model is correct roughly 74% of the time, so its 7% more accurate than a dumb model, so it does add some value. It's also more accurate
# than our logistics regression model by 3%. 


# first argument is true values, second argument is predicted probabilities

# we use y_test and y_pred_prob
# we don't use y_pred_class, because it will yield incorrect results without generating an error
# roc_curve returns 3 objects: fpr, tpr, threshold
# fpr: false positive rate
# tpr: true positive rate


# AUC is the percentage of the ROC plot that is underneath the curve:
# Higher value = better 
# Useful even when there is high class imbalance
auc = roc_auc_score(y1_test, y1_pred_prob)
print('AUC: %.2f' % auc)
# AUC 74%

# Receiver Operating Characteristic (ROC) curve can help you to choose a threshold that balances sensitivity and specificity
# Q: How are sensitivity and specificity affected by various thresholds
fpr, tpr, thresholds = roc_curve(y1_test, y1_pred_prob)
plt.plot([0, 1], [0, 1], linestyle='--') #no skill
plt.plot(fpr, tpr, label='ROC Curve with AUC area of %0.2f' % auc)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.0])
plt.rcParams['font.size'] = 12
plt.title('ROC curve for sex classifier')
plt.xlabel('False Positive Rate (1 - Specificity)')
plt.ylabel('True Positive Rate (Sensitivity)')
plt.legend(loc="lower right")
plt.grid(True)

# Accuracy Summary
# The Random Forest Classifier model has an AUC of .74, and is thus accurate 74% of the time.
# The model is incorrect 26% of the time. 
# When the actual value is positive, the prediction is correct 44% of the time.
# When the actual value is negative, the prediction is correct 88% of the time. 
# When the actual value is negative, the prediction is incorrect 12% of the time. 
# When a positive value is predicted, the prediction is correct 65% of the time. 
# F1 Score of 52%, which takes both precision and sensitivity into account.


# The logistic regression model has an AUC of .75, and is thus accurate 75% of the time.
# The model is incorrect 29% of the time. 
# When the actual value is positive, the prediction is correct 58% of the time.
# When the actual value is negative, the prediction is correct 77% of the time. 
# When the actual value is negative, the prediction is incorrect 23% of the time. 
# When a positive value is predicted, the prediction is correct 56% of the time. 
# F1 Score of 57%, which takes both precision and sensitivity into account.




# Conclusion
# the logistic regression model has a slightly higher AUC. But as with all models, there
# are pros and cons to everything, and it really depends what you're looking for in your
# model. As we see above, the differences in the percentages above. If we are aiming for a 
# model that makes fewer errors, the Random Forest Classifier is best. But if we're looking
# for a more well balanced model between precision and sensitivity, the Logistic Regression
# model is best. 




























