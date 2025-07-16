import numpy as np
import pandas as pd
import json
import time
import random
import sys
import os  
from queue import Queue
import pathlib

from math import ceil
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from sklearn import metrics


# fit the tree using gradient boosted classifier
def fit_boosted_tree(X, y, n_est=10, lr=0.1, d=3):
    clf = GradientBoostingClassifier( learning_rate=lr, n_estimators=n_est, max_depth=d,
                                    random_state=42)
    clf.fit(X, y)
    out = clf.score(X, y)
    return clf, out


# perform cut on the dataset

def cut(X, ts):
    df = X.copy()
    colnames = X.columns
    new_columns = []

    for j in range(len(ts)):
        for s in range(len(ts[j])):
            new_column_name = colnames[j] + '<=' + str(ts[j][s])
            new_column_data = np.where(df[colnames[j]] <= ts[j][s], 1, 0)
            new_columns.append(pd.DataFrame({new_column_name: new_column_data}, index=X.index))

    result = pd.concat(new_columns, axis=1)
    return result

# compute the thresholds
def get_thresholds(X, y, n_est, lr, d, backselect=True):
    # got a complaint here...
    y = np.ravel(y)
    # X is a dataframe
    clf, out = fit_boosted_tree(X, y, n_est, lr, d)
    # print('acc:', out)
    # assert 1==2
    thresholds = []
    for j in range(X.shape[1]):
        tj = np.array([])
        for i in range(len(clf.estimators_)):
            f = clf.estimators_[i,0].tree_.feature
            t = clf.estimators_[i,0].tree_.threshold
            tj = np.append(tj, t[f==j])
            # print(f, t, tj)
            # assert 1==2
        tj = np.unique(tj)
        thresholds.append(tj.tolist())
    # print(thresholds)
    # assert 1==2
    X_new = cut(X, thresholds)
    clf1, out1 = fit_boosted_tree(X_new, y, n_est, lr, d)
    # print('acc','1:', out1)#, 'acc1 cv:', scorep.mean())

    outp = 1
    Xp = X_new.copy()
    clfp = clf1
    itr=0
    vi = clfp.feature_importances_

    mask = vi >= 1e-4

    Xp = Xp.iloc[:, mask]

    if backselect:

        while outp >= min(0.91,out1)  and itr < X_new.shape[1]-1:
          
            vi = vi[mask]

      
            if vi.size > 0:
                c = Xp.columns
                i = np.argmin(vi)
                Xp = Xp.drop(c[i], axis=1)
                clfp, outp = fit_boosted_tree(Xp, y, n_est, lr, d)
                vi = clfp.feature_importances_
                itr += 1
            else:
                break
            
        Xp[c[i]] = X_new[c[i]]
 
    h = Xp.columns

    return Xp, thresholds, h

# compute the thresholds
def compute_thresholds(X, y, n_est, max_depth) :
    # n_est, max_depth: GBDT parameters
    # set LR to 0.1
    lr = 0.1
    start = time.perf_counter()
    X, thresholds, header = get_thresholds(X, y, n_est, lr, max_depth, backselect=True)
    guess_time = time.perf_counter()-start

    return X, thresholds, header, guess_time








