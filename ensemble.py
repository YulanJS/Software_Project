# To support both python 2 and python 3
from __future__ import division, print_function, unicode_literals

# Common imports
import numpy as np
# import os

# to make the output stable across runs
np.random.seed(42)

# To plot pretty figures
# import matplotlib
import matplotlib.pyplot as plt
plt.rcParams['axes.labelsize'] = 14  # meaning of x variable or y variable in English
plt.rcParams['xtick.labelsize'] = 12
plt.rcParams['ytick.labelsize'] = 12

# Where to save the figures
# PROJECT_ROOT_DIR = "C:/Users/jahan/Documents/SaintMaryCollege/Courses/OPS808/Summer 2018/PythonCodeExamples/figures"
# CHAPTER_ID = "decision_trees"
#
# def image_path(fig_id):
#    #return os.path.join(PROJECT_ROOT_DIR, "images", CHAPTER_ID, fig_id)
#    return os.path.join(PROJECT_ROOT_DIR, fig_id)
#
# def save_fig(fig_id, tight_layout=True):
#    print("Saving figure", fig_id)
#    if tight_layout:
#        plt.tight_layout()
#    plt.savefig(image_path(fig_id) + ".png", format='png', dpi=300)


from sklearn.model_selection import train_test_split
# from sklearn.datasets import make_moons
import pandas as pd

"""
In this section we will predict whether a bank note is authentic or fake 
depending upon the four different attributes of the image of the note. 
The attributes are Variance of wavelet transformed image, 
Kurtosis of the image, entropy, and skewness of the image.
"""
dataset = pd.read_csv("pima-indians-diabetes.data.csv", header=None)  # to point to data

print(dataset.shape)
print(dataset.head(20))

# drop columns without names: col index starts from 0
# df.drop(df.columns[0, 4, 6], axis=1)
X = dataset.drop(dataset.columns[8], axis=1)
print(X)
y = dataset[8]
print(y)

"""
Preparing the Data
In this section we will divide our data into attributes and labels 
and will then divide the resultant data into both training and test sets. 
By doing this we can train our algorithm on one set of data and then test it out 
on a completely different set of data that the algorithm hasn't seen yet. 
This provides you with a more accurate view of how your trained 
algorithm will actually perform. We now split the data
"""
# split into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20)
# Bagging ensembles

from sklearn.ensemble import BaggingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

# Bagging:
# n_estimators: combine the predictions of n_estimators base estimators
# max_samples: max number of samples to train each base estimator
# bootstrap: replacement or not
# n_jobs: num of jobs to run in parallel
bag_clf = BaggingClassifier(
    DecisionTreeClassifier(random_state=42), n_estimators=500,
    max_samples=100, bootstrap=True, n_jobs=1, random_state=42)

bag_clf.fit(X_train, y_train)
y_pred = bag_clf.predict(X_test)

# determine accuracy score for the bagging method
print("accuracy score for the bagging method: ")
print(accuracy_score(y_test, y_pred))

# now use a standard decision tree classifier
tree_clf = DecisionTreeClassifier(random_state=42)
tree_clf.fit(X_train, y_train)

y_pred_tree = tree_clf.predict(X_test)

print("accuracy score for standard decision tree classifier")
print(accuracy_score(y_test, y_pred))
# compare bagging method with standard decision tree classifier
# They almost have the same accuracy, which means that data set is already clean enough.

# Random Forests

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_curve, auc

# max_leaf_nodes
# no limitation on max_depth, split according to other parameters
rnd_clf = RandomForestClassifier(n_estimators=500, max_leaf_nodes=16, n_jobs=-1, random_state=42)
rnd_clf.fit(X_train, y_train)

y_prob_rf = rnd_clf.predict_proba(X_test)
print("Random Forest Classifier: y_prob_rf")
print(y_prob_rf)

y_pred_rf = rnd_clf.predict(X_test)

print("Compare random forest predictions with bagging predictions:")
print(np.sum(y_pred == y_pred_rf) / len(y_pred))  # almost identical predictions
# random forest predictions are almost same as the predictions of bagging.
# The performance of random forest is similar to the performance of bagging or decision tree classifier.

y_score_rf = y_prob_rf[:, 1]
# y = 1 means positive, to be used when analyze roc using false positive(x axis) and true positive(y axis)
print("Random Forest: y_score_rf")
print(y_score_rf)
fpr_rf, tpr_rf, threshold_rf = roc_curve(y_test, y_score_rf)


# ROC curve plotting
def plot_roc_curve(fpr, tpr, lable=None):
    # be careful of the curve around top left  corner
    plt.plot(fpr, tpr, linewidth=2)
    # pyplot.plot(x, y): x, y can be array-like, (0, 0) and (1, 1)
    plt.plot([0, 1], [0, 1], 'k--')
    # axis([xmin, xmax, ymin, ymax])
    plt.axis([0, 1, 0, 1])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    print("area under roc curve: ")
    print(auc(fpr, tpr))


plot_roc_curve(fpr_rf, tpr_rf)
plt.show()
# Random forest does a good job in distinguishing two categories.

# Out-of-Bag evaluation

# max_samples = 1.0 by default, 1 sample in each base(bootstrapped) tree
# oob_score=True. Whether to use out-of-bag samples to estimate the generalization error.
bag_clf = BaggingClassifier(
    DecisionTreeClassifier(random_state=42), n_estimators=500,
    bootstrap=True, oob_score=True, random_state=40)
bag_clf.fit(X_train, y_train)
print("Out of bag evaluation: ")
print(bag_clf.oob_score_)

from sklearn.metrics import accuracy_score
y_pred = bag_clf.predict(X_test)
print("bagging accuracy_score: ")
print(accuracy_score(y_test, y_pred))


# Boosting method, three boosting algorithms
# First Ada boost
from sklearn.ensemble import AdaBoostClassifier

ada_clf = AdaBoostClassifier(
    DecisionTreeClassifier(max_depth=1), n_estimators=200,
    algorithm="SAMME.R", learning_rate=0.5, random_state=42)
ada_clf.fit(X_train, y_train)

y_pred_ada = ada_clf.predict(X_test)
print("Ada Boost accuracy_score")
print(accuracy_score(y_test, y_pred_ada))
# Ada Boost almost has the same accuracy as decision tree classifier.

# compare random forest with different max_depth
# randome forest classifier using max_depth as parameter to calculate false positive rate and true positive rate


def random_forest_roc_analyze(depth):
    rnd_d_clf = RandomForestClassifier(n_estimators=500, max_depth=depth, max_leaf_nodes=16, n_jobs=-1, random_state=42)
    rnd_d_clf.fit(X_train, y_train)

    y_prob_rf_d = rnd_d_clf.predict_proba(X_test)

    y_pred_rf_d = rnd_d_clf.predict(X_test)

    print("Compare random forest predictions with bagging predictions: max_depth = {}".format(depth))
    print(np.sum(y_pred == y_pred_rf_d) / len(y_pred))  # almost identical predictions

    y_score_rf_d = y_prob_rf_d[:, 1]
    fpr_rf_d, tpr_rf_d, threshold_rf_d = roc_curve(y_test, y_score_rf_d)
    return fpr_rf_d, tpr_rf_d


# ROC curve plotting
def plot_roc_curve_mutiple(fpr_list, tpr_list, lable=None):
    # be careful of the curve around top left  corner
    fig, ax = plt.subplots()
    ax.plot(fpr_list[0], tpr_list[0], 'r--', label='max_depth=2')
    ax.plot(fpr_list[1], tpr_list[1], 'b--', label='max_depth=3')
    ax.plot(fpr_list[2], tpr_list[2], 'g--', label='max_depth=4')
    # pyplot.plot(x, y): x, y can be array-like, (0, 0) and (1, 1)
    ax.plot([0, 1], [0, 1], 'k--')
    # axis([xmin, xmax, ymin, ymax])
    plt.axis([0, 1, 0, 1])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    leg = ax.legend()


fpr_list, tpr_list = zip(random_forest_roc_analyze(2), random_forest_roc_analyze(3), random_forest_roc_analyze(4))
plot_roc_curve_mutiple(fpr_list, tpr_list)
plt.show()

# By plotting, we can see that there is not obvious difference in performance when max_depth is set to 2, 3 or 4.
"""
output:
(768, 9)
     0    1   2   3    4     5      6   7  8
0    6  148  72  35    0  33.6  0.627  50  1
1    1   85  66  29    0  26.6  0.351  31  0
2    8  183  64   0    0  23.3  0.672  32  1
3    1   89  66  23   94  28.1  0.167  21  0
4    0  137  40  35  168  43.1  2.288  33  1
5    5  116  74   0    0  25.6  0.201  30  0
6    3   78  50  32   88  31.0  0.248  26  1
7   10  115   0   0    0  35.3  0.134  29  0
8    2  197  70  45  543  30.5  0.158  53  1
9    8  125  96   0    0   0.0  0.232  54  1
10   4  110  92   0    0  37.6  0.191  30  0
11  10  168  74   0    0  38.0  0.537  34  1
12  10  139  80   0    0  27.1  1.441  57  0
13   1  189  60  23  846  30.1  0.398  59  1
14   5  166  72  19  175  25.8  0.587  51  1
15   7  100   0   0    0  30.0  0.484  32  1
16   0  118  84  47  230  45.8  0.551  31  1
17   7  107  74   0    0  29.6  0.254  31  1
18   1  103  30  38   83  43.3  0.183  33  0
19   1  115  70  30   96  34.6  0.529  32  1
      0    1   2   3    4     5      6   7
0     6  148  72  35    0  33.6  0.627  50
1     1   85  66  29    0  26.6  0.351  31
2     8  183  64   0    0  23.3  0.672  32
3     1   89  66  23   94  28.1  0.167  21
4     0  137  40  35  168  43.1  2.288  33
..   ..  ...  ..  ..  ...   ...    ...  ..
763  10  101  76  48  180  32.9  0.171  63
764   2  122  70  27    0  36.8  0.340  27
765   5  121  72  23  112  26.2  0.245  30
766   1  126  60   0    0  30.1  0.349  47
767   1   93  70  31    0  30.4  0.315  23

[768 rows x 8 columns]
0      1
1      0
2      1
3      0
4      1
      ..
763    0
764    0
765    0
766    1
767    0
Name: 8, Length: 768, dtype: int64
accuracy score for the bagging method: 
0.7662337662337663
accuracy score for standard decision tree classifier
0.7662337662337663
Random Forest Classifier: y_prob_rf
[[0.52590171 0.47409829]
 [0.82906326 0.17093674]
 [0.84070952 0.15929048]
 [0.74379255 0.25620745]
 [0.5324944  0.4675056 ]
 [0.44543812 0.55456188]
 [0.93621697 0.06378303]
 [0.42052252 0.57947748]
 [0.41089161 0.58910839]
 [0.41709186 0.58290814]
 [0.72691855 0.27308145]
 [0.25640361 0.74359639]
 [0.60684037 0.39315963]
 [0.59849438 0.40150562]
 [0.91262785 0.08737215]
 [0.63142916 0.36857084]
 [0.83675717 0.16324283]
 [0.89433343 0.10566657]
 [0.45084249 0.54915751]
 [0.57960091 0.42039909]
 [0.68395689 0.31604311]
 [0.83857289 0.16142711]
 [0.62389015 0.37610985]
 [0.89573222 0.10426778]
 [0.4593865  0.5406135 ]
 [0.21696902 0.78303098]
 [0.88832211 0.11167789]
 [0.91736161 0.08263839]
 [0.82140183 0.17859817]
 [0.75233569 0.24766431]
 [0.31501699 0.68498301]
 [0.36718124 0.63281876]
 [0.26481131 0.73518869]
 [0.25825148 0.74174852]
 [0.46578579 0.53421421]
 [0.28104356 0.71895644]
 [0.22120102 0.77879898]
 [0.52037872 0.47962128]
 [0.76314859 0.23685141]
 [0.45088519 0.54911481]
 [0.88865739 0.11134261]
 [0.70993653 0.29006347]
 [0.47966738 0.52033262]
 [0.55894822 0.44105178]
 [0.89182592 0.10817408]
 [0.38904588 0.61095412]
 [0.43584079 0.56415921]
 [0.79535392 0.20464608]
 [0.76360073 0.23639927]
 [0.18950707 0.81049293]
 [0.92914528 0.07085472]
 [0.25756901 0.74243099]
 [0.29711106 0.70288894]
 [0.7828122  0.2171878 ]
 [0.8620177  0.1379823 ]
 [0.90610473 0.09389527]
 [0.4370943  0.5629057 ]
 [0.92079834 0.07920166]
 [0.76963184 0.23036816]
 [0.28904554 0.71095446]
 [0.33148944 0.66851056]
 [0.77947629 0.22052371]
 [0.65248198 0.34751802]
 [0.6495844  0.3504156 ]
 [0.87070216 0.12929784]
 [0.44427127 0.55572873]
 [0.9161156  0.0838844 ]
 [0.43465973 0.56534027]
 [0.90170667 0.09829333]
 [0.30041107 0.69958893]
 [0.28805616 0.71194384]
 [0.79346035 0.20653965]
 [0.84527641 0.15472359]
 [0.88890654 0.11109346]
 [0.85250415 0.14749585]
 [0.58189669 0.41810331]
 [0.78871508 0.21128492]
 [0.82753378 0.17246622]
 [0.83350062 0.16649938]
 [0.74143145 0.25856855]
 [0.29722643 0.70277357]
 [0.85259133 0.14740867]
 [0.79974894 0.20025106]
 [0.51357026 0.48642974]
 [0.7957542  0.2042458 ]
 [0.20351002 0.79648998]
 [0.3417604  0.6582396 ]
 [0.59603796 0.40396204]
 [0.78000773 0.21999227]
 [0.88304975 0.11695025]
 [0.92207875 0.07792125]
 [0.76867293 0.23132707]
 [0.92191739 0.07808261]
 [0.53785488 0.46214512]
 [0.69567494 0.30432506]
 [0.4706472  0.5293528 ]
 [0.50079699 0.49920301]
 [0.87233467 0.12766533]
 [0.34974884 0.65025116]
 [0.73345734 0.26654266]
 [0.26625806 0.73374194]
 [0.87858136 0.12141864]
 [0.4746534  0.5253466 ]
 [0.44844296 0.55155704]
 [0.27665293 0.72334707]
 [0.75590575 0.24409425]
 [0.78845037 0.21154963]
 [0.20692994 0.79307006]
 [0.75618161 0.24381839]
 [0.36745504 0.63254496]
 [0.91017353 0.08982647]
 [0.53490078 0.46509922]
 [0.83879031 0.16120969]
 [0.2038295  0.7961705 ]
 [0.76883166 0.23116834]
 [0.76075984 0.23924016]
 [0.36793662 0.63206338]
 [0.6701556  0.3298444 ]
 [0.91622516 0.08377484]
 [0.56265004 0.43734996]
 [0.92431643 0.07568357]
 [0.79297245 0.20702755]
 [0.7008349  0.2991651 ]
 [0.88755764 0.11244236]
 [0.75734118 0.24265882]
 [0.53248813 0.46751187]
 [0.79904725 0.20095275]
 [0.23999068 0.76000932]
 [0.32805675 0.67194325]
 [0.37796797 0.62203203]
 [0.43803638 0.56196362]
 [0.28729751 0.71270249]
 [0.89807959 0.10192041]
 [0.56360888 0.43639112]
 [0.30394897 0.69605103]
 [0.72165884 0.27834116]
 [0.79029623 0.20970377]
 [0.32028052 0.67971948]
 [0.28503815 0.71496185]
 [0.93404213 0.06595787]
 [0.92029505 0.07970495]
 [0.92463217 0.07536783]
 [0.78230768 0.21769232]
 [0.60032734 0.39967266]
 [0.83336854 0.16663146]
 [0.6543337  0.3456663 ]
 [0.73115347 0.26884653]
 [0.931398   0.068602  ]
 [0.61483348 0.38516652]
 [0.2894984  0.7105016 ]
 [0.79957105 0.20042895]
 [0.62979147 0.37020853]
 [0.62545908 0.37454092]
 [0.66725742 0.33274258]]
Compare random forest predictions with bagging predictions:
0.987012987012987
Random Forest: y_score_rf
[0.47409829 0.17093674 0.15929048 0.25620745 0.4675056  0.55456188
 0.06378303 0.57947748 0.58910839 0.58290814 0.27308145 0.74359639
 0.39315963 0.40150562 0.08737215 0.36857084 0.16324283 0.10566657
 0.54915751 0.42039909 0.31604311 0.16142711 0.37610985 0.10426778
 0.5406135  0.78303098 0.11167789 0.08263839 0.17859817 0.24766431
 0.68498301 0.63281876 0.73518869 0.74174852 0.53421421 0.71895644
 0.77879898 0.47962128 0.23685141 0.54911481 0.11134261 0.29006347
 0.52033262 0.44105178 0.10817408 0.61095412 0.56415921 0.20464608
 0.23639927 0.81049293 0.07085472 0.74243099 0.70288894 0.2171878
 0.1379823  0.09389527 0.5629057  0.07920166 0.23036816 0.71095446
 0.66851056 0.22052371 0.34751802 0.3504156  0.12929784 0.55572873
 0.0838844  0.56534027 0.09829333 0.69958893 0.71194384 0.20653965
 0.15472359 0.11109346 0.14749585 0.41810331 0.21128492 0.17246622
 0.16649938 0.25856855 0.70277357 0.14740867 0.20025106 0.48642974
 0.2042458  0.79648998 0.6582396  0.40396204 0.21999227 0.11695025
 0.07792125 0.23132707 0.07808261 0.46214512 0.30432506 0.5293528
 0.49920301 0.12766533 0.65025116 0.26654266 0.73374194 0.12141864
 0.5253466  0.55155704 0.72334707 0.24409425 0.21154963 0.79307006
 0.24381839 0.63254496 0.08982647 0.46509922 0.16120969 0.7961705
 0.23116834 0.23924016 0.63206338 0.3298444  0.08377484 0.43734996
 0.07568357 0.20702755 0.2991651  0.11244236 0.24265882 0.46751187
 0.20095275 0.76000932 0.67194325 0.62203203 0.56196362 0.71270249
 0.10192041 0.43639112 0.69605103 0.27834116 0.20970377 0.67971948
 0.71496185 0.06595787 0.07970495 0.07536783 0.21769232 0.39967266
 0.16663146 0.3456663  0.26884653 0.068602   0.38516652 0.7105016
 0.20042895 0.37020853 0.37454092 0.33274258]
area under roc curve: 
0.8310376492194673
Out of bag evaluation: 
0.760586319218241
bagging accuracy_score: 
0.7532467532467533
Ada Boost accuracy_score
0.7467532467532467
Compare random forest predictions with bagging predictions: max_depth = 2
0.8376623376623377
Compare random forest predictions with bagging predictions: max_depth = 3
0.8636363636363636 n
Compare random forest predictions with bagging predictions: max_depth = 4ll
0.8896103896103896 i

Process finished with exit code 0
"""