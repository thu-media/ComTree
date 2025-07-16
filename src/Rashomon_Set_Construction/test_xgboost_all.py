import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import export_graphviz
from sklearn.model_selection import train_test_split

from model.threshold_guess_xgboost import compute_thresholds, cut
import xgboost as xgb

import pickle as pk
import csv
import os
# import pensieve
# import pensiedt
from sklearn.metrics import accuracy_score
import pandas as pd


from multiprocessing import Pool
import time
import h5py
from model.gosdt import GOSDT
columns =  ['last_quality', 'curr_buffer', 'tput_0', 'tput_1', 'tput_2', 'tput_3', 'tput_4', 'tput_5', 'tput_6', 'tput_7', 'delay_0', 'delay_1', 'delay_2', 'delay_3', 'delay_4', 'delay_5', 'delay_6', 'delay_7', 'size_0', 'size_1', 'size_2', 'size_3', 'size_4', 'size_5', 'chunk_til_video_end']

feature_num = 25
def get_gosdt(X_train_guessed, X_test_guessed, y_train, y_test, lamda,model_name):

    config = {

                "regularization": lamda,
                  "worker_limit": 220,
                "depth_budget": 6,
                "feature_exchange":True,
                'time_limit':1200,
                "verbose": True,
                  "uncertainty_tolerance": 0.001,
                 "model":"file/"+str(model_name),
    }

    model = GOSDT(config)

    model.fit(X_train_guessed, pd.DataFrame(y_train))
    print("evaluate the model, extracting tree and scores\n")
    train_acc = model.score(X_train_guessed, y_train)
    test_acc = model.score(X_test_guessed, y_test)
    n_leaves = model.leaves()
    n_nodes = model.nodes()
    time = model.utime

    print(f"Model training time: {time}\n")
    print(f"Training accuracy: {train_acc}\n")
    print(f"Test accuracy: {test_acc}\n")
    print(f"# of leaves: {n_leaves}\n")


def get_gdbt(X_train, X_test, y_train, y_test,n_est,max_depth,with_guess):
    clf = GradientBoostingClassifier(n_estimators=n_est, max_depth=max_depth, random_state=42)
    clf.fit(X_train, y_train.values.flatten())
    out =  clf.score(X_train, y_train.values.flatten())
    print(f"with_guess:{with_guess}, GDBT train Accuracy: {out:.2f}\n")
    y_pred = clf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"with_guess:{with_guess}, GDBT test Accuracy: {accuracy:.2f}\n")
    if not with_guess:
        X_train_guessed, thresholds, header, threshold_guess_time = compute_thresholds(X_train.copy(), y_train, n_est, max_depth)
        X_test_guessed = cut(X_test.copy(), thresholds)
        X_test_guessed = X_test_guessed[header]
        return X_train_guessed, X_test_guessed,''
def get_xgboost(X_train, X_test, y_train, y_test,n_est,max_depth,with_guess):

    clf = xgb.XGBClassifier(n_estimators=n_est, max_depth=max_depth,n_jobs = 220,   random_state=42)
    clf.fit(X_train, y_train.values.flatten())
    out = clf.score(X_train, y_train.values.flatten())
    print(f"with_guess:{with_guess}, XGBoost train Accuracy: {out}\n")
    y_pred = clf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"with_guess:{with_guess}, XGBoost test Accuracy: {accuracy}\n")
    if not with_guess:
        X_train_guessed, thresholds, header, threshold_guess_time = compute_thresholds(X_train.copy(), y_train, n_est, max_depth)
        X_test_guessed = cut(X_test.copy(), thresholds)
        X_test_guessed = X_test_guessed[header]
        return X_train_guessed, X_test_guessed, ''
    return '', '', ''
import sys
item = sys.argv[1]
name = 'gosdtc'+item
df =h5py.File('save/datac'+item, 'r') 
X_train = pd.DataFrame(df['X_train'], columns=columns)
X_test = pd.DataFrame(df['X_test'], columns=columns)

# columns = X_train.columns.tolist()
# print(columns,type(columns))
# assert 1==2
y_train =  pd.Series(df['y_train'], name= 'test')
y_test =  pd.Series(df['y_test'], name= 'test')
df.close()
from sklearn.ensemble import GradientBoostingClassifier

n_est = 4
max_depth = 3
with_guess = 0
X_train_guessed, X_test_guessed,_  = get_xgboost(X_train, X_test, y_train, y_test,n_est,max_depth,with_guess)
print(f" {X_train_guessed.shape}, {X_test_guessed.shape}, {y_train.shape}, {y_test.shape}\n")
with_guess = 1
columns = X_train_guessed.columns.tolist()
with h5py.File("save/columnsc"+item, "w") as newf:
  newf['c'] = columns 
with h5py.File("save/datac_g"+item, "w") as f:
    f['X_train_guessed'] = X_train_guessed
    f['X_test_guessed'] = X_test_guessed
get_xgboost(X_train_guessed, X_test_guessed, y_train, y_test,n_est,max_depth,with_guess)

get_gosdt(X_train_guessed, X_test_guessed, y_train, y_test, 0.0005,name)

