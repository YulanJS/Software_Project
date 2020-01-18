# Multiple Linear Regression HW: Yulan Jin
import pandas as pd
import statsmodels.api as sm
from sklearn.linear_model import LinearRegression
from sklearn.feature_selection import RFE
import matplotlib.pyplot as plt
import seaborn as sns
from pandas.plotting import scatter_matrix
from pandas import set_option
from pandas import read_csv
import numpy as np

# 1.
df = pd.read_csv('Baseball_salary.csv')
data1 = df.drop(['Unnamed: 0'], axis=1)
# print(data1)


# 2.
# separate features and log of response into two dataframes
# exclude categorical columns
# remove rows with missing values
data1 = data1.drop(['League', 'Division', 'NewLeague'], axis=1).dropna()
# reset_index: index starts from zero, may be used for changing format in sm.add_constant
data = data1.reset_index(drop=True)
X1 = data.drop(['Salary'], axis=1)
pd.set_option('display.max_columns', 100)
print('input: ')
print(X1)
# select a df with one col with data.iloc, apply np function, rename col name
Y1 = data.iloc[:, [16]].apply(np.log).rename(columns={'Salary': 'logSalary'})
pd.set_option('display.max_columns', 100)
print('output: ')
print(Y1)
# input and output in numpy array form
X = X1.values
# Y1.values will get shape(322, 1). Needs shape(322, ), which is a 1d numpy array
Y = Y1.values[:, 0]

# 3.
# recursive features elimination
NUM_FEATURES = 3  # this is kind of arbitrary but the idea should come by observing the scatter plots and correlation.
model = LinearRegression()
# recursive feature elimination: RFE
rfe = RFE(model, NUM_FEATURES)
# X Y are arrays
result = rfe.fit(X, Y)

print("Num Features:", result.n_features_)
print("Selected Features:", result.support_)
print("Feature Ranking:", result.ranking_)
# calculate the score for the selected features
score = rfe.score(X, Y)
print("Model Score with selected features is: ", score)

"""
when NUM_FEATURES = 16, score = 0.5426879134785904. Choose all features.
when NUM_FEATURES = 15, score = 0.5421346628328056. Eliminate features CAtBat because score doesn't change much.
when NUM_FEATURES = 14, score = 0.5421346614038307. 
Eliminate features CAtBat and CHmRun because score doesn't change much.
when NUM_FEATURES = 13, score = 0.5420665933153362.
Eliminate features CAtBat, CHits and CHmRun because score doesn't change much.
when NUM_FEATURES = 12, score = 0.5420009320369483.
Eliminate features CAtBat, CHits, CHmRun and CRBI because score doesn't change much.
when NUM_FEATURES = 11, score = 0.5419998291833658.
Eliminate features RBI, CAtBat, CHits, CHmRun and CRBI because score doesn't change much.
when NUM_FEATURES = 10, score = 0.5314183902202346.
Eliminate features RBI, CAtBat, CHits, CHmRun, CRBI and PutOuts because score doesn't change much.
when NUM_FEATURES = 9, score = 0.5289460050145818.
Eliminate features RBI, CAtBat, CHits, CHmRun, CRBI, PutOuts and Assists because score doesn't change much.
when NUM_FEATURES = 8, score = 0.520033425813008.
Eliminate features RBI, CAtBat, CHits, CHmRun, CRBI, CWalk, PutOuts and Assists because score doesn't change much.
when NUM_FEATURES = 7, score = 0.5111769290446959.
Eliminate features RBI, CAtBat, CHits, CRuns, CHmRun, CRBI, CWalk, PutOuts and Assists 
because score doesn't change much.
when NUM_FEATURES = 6, score = 0.5111753761077804.
Eliminate features Runs, RBI, CAtBat, CHits, CRuns, CHmRun, CRBI, CWalk, PutOuts and Assists 
because score doesn't change much.
when NUM_FEATURES = 5, score = 0.5021502314764965.
Eliminate features AtBat, Runs, RBI, CAtBat, CHits, CRuns, CHmRun, CRBI, CWalk, PutOuts and Assists 
because score doesn't change much.
when NUM_FEATURES = 4, score = 0.5015297981210649.
Eliminate features AtBat, HmRun, Runs, RBI, CAtBat, CHits, CRuns, CHmRun, CRBI, CWalk, PutOuts and Assists 
because score doesn't change much.
when NUM_FEATURES = 3, score = 0.48623120534368935.
Eliminate features AtBat, HmRun, Runs, RBI, Walks, CAtBat, CHits, CRuns, CHmRun, CRBI, CWalk, PutOuts and Assists 
because score doesn't change much.
when NUM_FEATURES = 2, score = 0.29288078159497144. We have to stop here because the score drops dramatically.
Conclusion:
Only keeps Hits, Years, and Errors in the final model. 
"""

# stepwise forward-backward selection
# need to change the input types as X in this function needs to be a pandas DataFrame


def stepwise_selection(X, y,
                       initial_list=[],
                       threshold_in=0.01,
                       threshold_out=0.05,
                       verbose=True):
    """ Perform a forward-backward feature selection
    based on p-value from statsmodels.api.OLS
    Arguments:
        X - pandas.DataFrame with candidate features
        y - list-like with the target
        initial_list - list of features to start with (column names of X)
        threshold_in - include a feature if its p-value < threshold_in
        threshold_out - exclude a feature if its p-value > threshold_out
        verbose - whether to print the sequence of inclusions and exclusions
    Returns: list of selected features
    Always set threshold_in < threshold_out to avoid infinite looping.
    See https://en.wikipedia.org/wiki/Stepwise_regression for the details
    """
    included = list(initial_list)
    while True:
        changed = False
        # forward step
        excluded = list(set(X.columns) - set(included))
        new_pval = pd.Series(index=excluded)
        for new_column in excluded:
            # sm.add_constant: add a column of ones to the array, by default, adding ones as the first column
            # Multiple linear regression, X matrix: first column is a column of ones
            # X[a list of columns] to extract columns
            # change sample code just because Method.php will be deprecated, use numpy.php instead
            # has to use dataframe.values as parameter for sm.add_constant, will also return numpy array
            # To keep input parameter of OLS as dataframe,
            # convert numpy array result from sm.add_constant() to dataframe
            model = sm.OLS(y, pd.DataFrame(sm.add_constant(X[included + [new_column]].values),
                                           columns=['const'] + X[included + [new_column]].columns.tolist())).fit()
            # model.pvalues: p values of t statistics
            new_pval[new_column] = model.pvalues[new_column]
        best_pval = new_pval.min()
        if best_pval < threshold_in:
            # argmin() return row label of min values
            best_feature = new_pval.idxmin()
            included.append(best_feature)
            changed = True
            if verbose:
                print('Add  {:30} with p-value {:.6}'.format(best_feature, best_pval))

        # backward step
        model = sm.OLS(y, pd.DataFrame(sm.add_constant(X[included].values),
                                       columns=['const'] + X[included].columns.tolist())).fit()
        # use all coefs except intercept
        # first row is the p value for const(ones)
        pvalues = model.pvalues.iloc[1:]
        worst_pval = pvalues.max()  # null if pvalues is empty
        if worst_pval > threshold_out:
            changed = True
            worst_feature = pvalues.idxmax()
            included.remove(worst_feature)
            if verbose:
                print('Drop {:30} with p-value {:.6}'.format(worst_feature, worst_pval))
        if not changed:
            break
    return included


resulting_features = stepwise_selection(X1, Y1)

print('resulting features:')
print(resulting_features)

# p value is the performance measure for stepwise feature selection.
# However, this method doesn't take into account of RSS(or score), which should also be the performance measure.
# For the stepwise feature selection method, the final model has features: CRuns, Hits and Years.
"""

input: 
     AtBat  Hits  HmRun  Runs  RBI  Walks  Years  CAtBat  CHits  CHmRun  \
0      315    81      7    24   38     39     14    3449    835      69   
1      479   130     18    66   72     76      3    1624    457      63   
2      496   141     20    65   78     37     11    5628   1575     225   
3      321    87     10    39   42     30      2     396    101      12   
4      594   169      4    74   51     35     11    4408   1133      19   
..     ...   ...    ...   ...  ...    ...    ...     ...    ...     ...   
258    497   127      7    65   48     37      5    2703    806      32   
259    492   136      5    76   50     94     12    5511   1511      39   
260    475   126      3    61   43     52      6    1700    433       7   
261    573   144      9    85   60     78      8    3198    857      97   
262    631   170      9    77   44     31     11    4908   1457      30   

     CRuns  CRBI  CWalks  PutOuts  Assists  Errors  
0      321   414     375      632       43      10  
1      224   266     263      880       82      14  
2      828   838     354      200       11       3  
3       48    46      33      805       40       4  
4      501   336     194      282      421      25  
..     ...   ...     ...      ...      ...     ...  
258    379   311     138      325        9       3  
259    897   451     875      313      381      20  
260    217    93     146       37      113       7  
261    470   420     332     1314      131      12  
262    775   357     249      408        4       3  

[263 rows x 16 columns]
output: 
     logSalary
0     6.163315
1     6.173786
2     6.214608
3     4.516339
4     6.620073
..         ...
258   6.551080
259   6.774224
260   5.953243
261   6.866933
262   6.907755

[263 rows x 1 columns]
Num Features: 16
Selected Features: [ True  True  True  True  True  True  True  True  True  True  True  True
  True  True  True  True]
Feature Ranking: [1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1]
Model Score with selected features is:  0.5426879134785904
Add  CRuns                          with p-value 1.88938e-29
Add  Hits                           with p-value 2.94819e-11
Add  Years                          with p-value 0.00578972
resulting features:
['CRuns', 'Hits', 'Years']

Process finished with exit code 0

"""