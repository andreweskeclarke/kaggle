import time
import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from sklearn import decomposition
from sklearn import preprocessing
from sklearn import cluster
from sklearn import linear_model
from sklearn import ensemble
from sklearn import cross_validation
from sklearn.metrics import mean_absolute_error



def get_actual_y(data):
    return data.groupby('Id').mean()[['Expected']]

def simplest_predictions(train, test):
    # Build simplest model for reference
    median_predictions = get_actual_y(test)
    median_predictions['Expected'] = train['Expected'].median()
    return median_predictions

def predict(train, test):
    predictions = get_actual_y(test)
    predictions['Expected'] = train['Expected'].median()
    train = prep_and_filter_data(train)
    test = prep_and_filter_data(test)

    # Full tree
    print("Full tree...")
    full_tree_train_data = train[train.count(1) == 9]
    full_tree_test_data = test[test.count(1) == 9]
    print("Full tree on {}...".format(full_tree_train_data.shape))
    model = ensemble.RandomForestRegressor(n_estimators=N_EST, max_depth=MAX_D, n_jobs=-1, min_samples_split=MIN_LEAF, max_features=MAX_FEATURES)
    full_tree_test_data['predictions'] = model.fit(X=full_tree_train_data[full_tree_train_data.columns.difference(['Id','Expected'])], y=full_tree_train_data['Expected']).predict(X=full_tree_test_data[full_tree_test_data.columns.difference(['Id','Expected'])])


    # Tree with only means
    print("Partial tree...")
    partial_tree_train_data = train[train.count(1) < 9][train['Ref_mean'].notnull()][train['RhoHV_mean'].notnull()][train['Zdr_mean'].notnull()][train['Kdp_mean'].notnull()]
    partial_tree_train_data = partial_tree_train_data.loc[:,['Ref_mean','RhoHV_mean','Zdr_mean','Kdp_mean','Expected']].copy()
    partial_tree_test_data = test[test.count(1) < 9][test['Ref_mean'].notnull()][test['RhoHV_mean'].notnull()][test['Zdr_mean'].notnull()][test['Kdp_mean'].notnull()]
    partial_tree_test_data = partial_tree_test_data.loc[:,['Ref_mean','RhoHV_mean','Zdr_mean','Kdp_mean','Expected']].copy()
    print("Partial tree on {}...".format(partial_tree_train_data.shape))
    partial_model = ensemble.RandomForestRegressor(n_estimators=N_EST, max_depth=MAX_D, n_jobs=-1, min_samples_split=MIN_LEAF, max_features='auto')
    partial_tree_test_data['predictions'] = partial_model.fit(X=partial_tree_train_data[partial_tree_train_data.columns.difference(['Id','Expected'])], y=partial_tree_train_data['Expected']).predict(X=partial_tree_test_data[partial_tree_test_data.columns.difference(['Id','Expected'])])

    for i in predictions.index:
        if i in full_tree_test_data.index:
            predictions.loc[i,'Expected'] = full_tree_test_data.loc[i,'predictions']
        elif i in partial_tree_test_data.index:
            predictions.loc[i,'Expected'] = partial_tree_test_data.loc[i,'predictions']
    print(predictions)
    return predictions

def run(data):
    e_tot = 0
    med_e_tot = 0
    for t1, t2 in cross_validation.KFold(data.shape[0], n_folds=10, shuffle=True):
        # Prep data - still raw
        train = data.iloc[t1]
        test = data.iloc[t2]
        y = get_actual_y(test)
        
        e = error_rate(y['Expected'], predict(train, test)['Expected'])
        med_e = error_rate(y['Expected'], simplest_predictions(train, test)['Expected'])
        e_tot += e
        med_e_tot += med_e
        print("Median error rate: {} --- Error rate: {}".format(med_e, e))

    print("Avg median error: {}".format(med_e_tot / 10))
    print("Avg error: {}".format(e_tot / 10))

def error_rate(expected, predicted):
    # MAE
    return (expected - predicted).abs().mean()

def prep_and_filter_data(data):
    means = data.groupby('Id').mean()
    means.columns += '_mean'
    stds = data.groupby('Id').std()
    stds.columns += '_std'
    comb = pd.concat([stds, means], axis=1)
    comb.drop('Expected_std', axis=1, inplace=True)
    comb = comb[comb['Ref_mean'] > 0]
    comb = comb[comb['Expected_mean'] < 70]
    comb['Expected'] = comb['Expected_mean']
    comb.drop('Expected_mean', inplace=True, axis=1)
    return comb

# Data + features
# data_raw = pd.read_csv('input/train_clean.csv', usecols=[0,3,11,15,19,23])
MAX_FEATURES='auto'; N_EST=30; MAX_D=None; MIN_LEAF=5000;
run(data_raw)


# train_raw = pd.read_csv('input/train_clean.csv', usecols=[0,3,11,15,19,23])
# test_raw = pd.read_csv('input/test.csv', usecols=[0,3,11,15,19])
predict(train_raw, test_raw)
