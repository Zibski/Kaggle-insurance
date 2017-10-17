'''
Created on 12 paz 2017

@author: Zbyszek
'''
import pandas as pd
import numpy as np
from catboost import CatBoostClassifier
from sklearn.model_selection import train_test_split
import os
from datetime import datetime
import hyperopt

import data_loader
import gini

nrep = 100
os.chdir('C:\kaggle\Porto Seguro')

data_files = data_loader.load_all_data_files()
test_df = data_loader.load_data(data_files[1])
train_df = data_loader.load_data(data_files[2])

print("Train: ", train_df.shape)
print("Test: ", test_df.shape)

exclude_missing = []
exclude_other = ['id', 'target']
exclude_unique = []

train_features = []
for c in train_df.columns:
    if c not in exclude_missing \
       and c not in exclude_other and c not in exclude_unique:
        train_features.append(c)
print("We use these for training: %s" % train_features)
print(len(train_features))

cat_feature_inds = []
for i, c in enumerate(train_features):
    num_uniques = len(train_df[c].unique())
    if 'cat' in c:
        cat_feature_inds.append(i)
        
print("Cat features are: %s" % [train_features[ind] for ind in cat_feature_inds])

X_train, X_test, y_train, y_test = train_test_split(train_df[train_features], train_df.target, test_size=0.33, random_state=8)
#X_train = train_df[train_features]
#y_train = train_df.target
#X_test = test_df[train_features]
print('Test shape: {}'.format( X_test.shape))
print(X_train.shape, y_train.shape)

space_hyperopt = {
    'iterations': hyperopt.hp.quniform('iterations', 300, 1000, 100),
    'learning_rate': hyperopt.hp.quniform('learning_rate', 0.02, 0.1, 0.001),
    'depth': hyperopt.hp.quniform('depth', 5, 16, 1),
    'l2_leaf_reg': hyperopt.hp.quniform('l2_leaf_reg', 1, 6, 1),
    'loss_function': 'Logloss'
    }


def clf_gini(clf, xtr, ytr, xte, yte):
    clf.fit(xtr, ytr)
    pred = clf.predict_proba(xte)
    return gini.gini_normalized(yte, pred)

def obj_func(hyperparams):
    # uses global xtr, ytr, xval, yval
    #hyperparams = xgb_parse_params(hyperparams)
    obj = clf_gini(CatBoostClassifier(**hyperparams), X_train, y_train, X_test, y_test)
    print('obj: ', '{0:.3f}'.format(obj), 'arguments', hyperparams, '\n')
    return obj

algo = hyperopt.tpe.suggest
trials = hyperopt.Trials() 
best = hyperopt.fmin(fn= obj_func,
            space=space_hyperopt,
            algo=algo,
            max_evals=nrep,
            trials=trials)

print('Best hyperopt: {}'.format(best))
print(list(trials)[0])

    
    

