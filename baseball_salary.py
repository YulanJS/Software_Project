# Hw on Exploratory Data Analysis: Yulan Jin

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pandas.plotting import scatter_matrix

# Question 2
# load the data
df = pd.read_csv('Baseball_salary.csv')
data = df.drop(['Unnamed: 0'], axis=1)
#print(data)

# separate features and log of response into two dataframes
X = data.drop(['League', 'Division', 'NewLeague', 'Salary'], axis=1)
pd.set_option('display.max_columns', 100)
print(X)
# select a df with one col with data.iloc, apply np function, rename col name
Y = data.iloc[:, [18]].apply(np.log).rename(columns={'Salary': 'logSalary'})
pd.set_option('display.max_columns', 100)
print(Y)

# Question 3
# correlation, temporarily using logSalary
plt.figure()
preprocessed_data = pd.concat([X, Y], axis=1)
corrMat = preprocessed_data.corr(method='pearson')
pd.set_option('display.max_columns', 100)
print(corrMat)
sns.heatmap(corrMat, square='True')
plt.yticks(rotation=0)
plt.xticks(rotation=90)
plt.title('Correlation Matrix Using Heat Map')
plt.show()
# candidates for the model: CAtBat, CHits, CRuns, CRBI, Years, CHmRun, CWalks
# from the correlation matrix, those features above have relatively higher correlation values than other features,
# which are between 0.4 and 0.6.
# The colors in the grids of heatmap also correspond to correlation values between 0.4 and 0.6.

# scatter point
scatter_matrix(preprocessed_data)
plt.show()
# From scatter point, these features seem to be closely related to logSalary:
# CAtBat, CHits, CHmRun, CRuns.
# LogSalary and each feature above probably have a linear relationship.
# CRBI, years, CWalks are relatively concentrated in the left upper part,
# each of them may have a linear relationship with logSalary.

# Therefore, according to correlation matrix and scatter point, CAtBat, CHits, CHmRun, CRuns, CRBI, Years, CWalks
# are candidate features for the model.

# Question 4
print(preprocessed_data.describe())
# This form shows some important statistical values.

#Question 5
preprocessed_data.hist()
plt.show()
# Histogram shows the general distribution of each feature.
# Most of features skewed right. For example, Asists, CAtBat, CHits, CHmRun, CRBI, CRuns, CWalks, Erros,
# HmRun, PutOuts, RBI, Runs, Walks, Years.
# Some features look normal, such as AtBat, Hits,
# LogSalary skewed left.
