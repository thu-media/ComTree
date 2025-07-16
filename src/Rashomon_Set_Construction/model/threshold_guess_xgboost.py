import numpy as np
import pandas as pd
import json
import time
import random
import sys
import os  
from queue import Queue
import pathlib
import xgboost as xgb
from math import ceil
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from sklearn import metrics


# fit the tree using gradient boosted classifier
def fit_boosted_tree(X, y, n_est=10, lr=0.1, d=1):

    clf = xgb.XGBClassifier(n_estimators=n_est, max_depth=d,n_jobs = 220, random_state=42)
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
            new_column_name = f"{colnames[j]}_less_equal_{ts[j][s]}"
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

    booster = clf.get_booster()

    trees_info = booster.get_dump(with_stats=True)

    thresholds = []
    for j in range(X.shape[1]):
        tj = np.array([])
        for tree_info in trees_info:
            for line in tree_info.split('\n'):
                # print(line)
                if f"[f{j}<" in line or f"[{X.columns[j]}<" in line:
                    parts = line.split("[")
                    for part in parts[1:]:
                        feature_name, threshold_str = part.split("<")
                        threshold = float(threshold_str.split("]")[0])
                
                        tj = np.append(tj, threshold)
        
   
        tj = np.unique(tj)
        thresholds.append(tj.tolist())
    

    X_new = cut(X, thresholds)
    clf1, out1 = fit_boosted_tree(X_new, y, n_est, lr, d)

    outp = 1
    Xp = X_new.copy()
    clfp = clf1
    itr=0
    # print(Xp.shape)
    vi = clfp.feature_importances_

    mask = vi >= 5e-3

    Xp = Xp.iloc[:, mask]

    vi = vi[mask]
    if backselect:

        while outp >= min(0.91,out1)  and itr < X_new.shape[1]-1:
          

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
    #print('features:', h)
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








