###3###

# Generating a synthetic data set
from sklearn import datasets

X, y = datasets.make_classification(
    weights=[0.8, 0.2],
    n_samples=100,
    n_features=2,
    n_informative=1,
    n_repeated=0,
    n_redundant=0,
    flip_y=.05,
    random_state=1234,
    n_clusters_per_class=1,
)

# Splitting the data into training and test sets
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.3,  # 30% goes to testing, 70% goes to training
    random_state=1234,
)

# Building the model

from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score

# Testing the model

clf = GaussianNB()
clf.fit(X_train, y_train)
predict = clf.predict(X_test)
score = accuracy_score(y_test, predict)
print("Simple accuracy score: %.2f" % score)

### CROSS VALIDATION ###

from sklearn.model_selection import KFold

kf = KFold(n_splits=5, shuffle=True, random_state=1234)  # 5 folds, shuffle them before before division

scores = []

for train_index, test_index in kf.split(X):
    # kf.split return index numbers of samples divided into test and train splits
    X_train, X_test = X[train_index], X[test_index]  # division into train and test atributes
    y_train, y_test = y[train_index], y[test_index]
    clf.fit(X_train, y_train)
    predict = clf.predict(X_test)
    scores.append(accuracy_score(y_test, predict))

import numpy as np
mean_score = np.mean(scores)    # mean value of scores
std_score =np.std(scores)       # standard deviation of scores
print("Cross validation accuracy score:: %.3f (%.3f)" % (mean_score, std_score))

print(f"y labels:\n{y}")
print(f"y label pretty:\n{np.unique(y, return_counts=True)}")


### STRATIFIED CROSS VALIDATION ###

from sklearn.model_selection import StratifiedKFold
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=1234)
scores = []

for train_index, test_index in skf.split(X,y):      # only different here!
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]
    clf.fit(X_train, y_train)
    predict = clf.predict(X_test)
    scores.append(accuracy_score(y_test, predict))

mean_score = np.mean(scores)
std_score = np.std(scores)
print("Stratified cross validation accuracy score: %.3f (%.3f)" % (mean_score, std_score))

### K-FOLD CROSS VALIDATION ###

from sklearn.model_selection import RepeatedKFold
rkf = RepeatedKFold(n_splits=5, n_repeats=10, random_state=1234) # different here
scores = []

for train_index, test_index in rkf.split(X):
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]
    clf.fit(X_train, y_train)
    predict = clf.predict(X_test)
    scores.append(accuracy_score(y_test, predict))

mean_score = np.mean(scores)
std_score = np.std(scores)
print("Accuracy score: %.3f (%.3f)" % (mean_score, std_score))


### STRATIFIED K-FOLD CROSS VALIDATION ###

from sklearn.model_selection import RepeatedStratifiedKFold
rskf = RepeatedStratifiedKFold(n_splits=5, n_repeats=10, random_state=1234) # name's different here
scores = []

for train_index, test_index in rskf.split(X,y):
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]
    clf.fit(X_train, y_train)
    predict = clf.predict(X_test)
    scores.append(accuracy_score(y_test, predict))

mean_score = np.mean(scores)
std_score = np.std(scores)
print("Accuracy score: %.3f (%.3f)" % (mean_score, std_score))

### LEAVE ONE OUT ### (used for small sets of data)

from sklearn.model_selection import LeaveOneOut
loo = LeaveOneOut()

for train_index, test_index in loo.split(X):
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]
    clf.fit(X_train, y_train)
    predict = clf.predict(X_test)
    scores.append(accuracy_score(y_test, predict))
mean_score = np.mean(scores)
std_score = np.std(scores)
print("Accuracy score: %.3f (%.3f)" % (mean_score, std_score))













