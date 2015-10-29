import statistics
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


# Kaggle example
def marshall_palmer(ref, minutes_past):
    #print "Estimating rainfall from {0} observations".format(len(minutes_past))
    # how long is each observation valid?
    valid_time = np.zeros_like(minutes_past)
    valid_time[0] = minutes_past.iloc[0]
    for n in range(1, len(minutes_past)):
        valid_time[n] = minutes_past.iloc[n] - minutes_past.iloc[n-1]
    valid_time[-1] = valid_time[-1] + 60 - np.sum(valid_time)
    valid_time = valid_time / 60.0

    # sum up rainrate * validtime
    sum = 0
    for dbz, hours in zip(ref, valid_time):
        # See: https://en.wikipedia.org/wiki/DBZ_(meteorology)
        if np.isfinite(dbz):
            mmperhr = pow(pow(10, dbz/10)/200, 0.625)
            sum = sum + mmperhr * hours
    return sum


# Kaggle example
# each unique Id is an hour of data at some gauge
def myfunc(hour):
    #rowid = hour['Id'].iloc[0]
    # sort hour by minutes_past
    hour = hour.sort('minutes_past', ascending=True)
    est = marshall_palmer(hour['Ref'], hour['minutes_past'])
    return est

def cluster_data(train_raw, test_raw):
    # Normalize before building PCA components
    cluster_size = 7
    train = train_raw.fillna(-1)
    test = test_raw.fillna(-1)
    train_norm = preprocessing.scale(train.loc[:,['Ref','RefComposite','RhoHV','Zdr','Kdp']])
    pca = decomposition.PCA(n_components=5).fit(train_norm)
    train_pca = pca.transform(train_norm)
    # Cluster measurements based on PCA components
    clusterer = cluster.KMeans(n_clusters=cluster_size, n_init=15, max_iter=300, init='k-means++').fit(train_pca)
    train_categories = clusterer.predict(train_pca)
    train_dummies = pd.get_dummies(train_categories)
    col_names = []
    for i in range(0,cluster_size):
        col_names.append('cat' + str(i))
    train_dummies.columns = col_names
    train_dummies.set_index(train.index, inplace=True)
    train_dummies['Id'] = train_raw['Id']
    train_raw = pd.concat([train_raw, train_dummies.drop('Id', axis=1)], axis=1) 

    test_norm = preprocessing.scale(test.loc[:,['Ref','RefComposite','RhoHV','Zdr','Kdp']])
    test_pca = pca.transform(test_norm)
    test_dummies = pd.get_dummies(clusterer.predict(test_pca))
    test_dummies.columns = col_names
    test_dummies.set_index(test.index, inplace=True)
    test_dummies['Id'] = test_raw['Id']
    test_raw = pd.concat([test_raw, test_dummies.drop('Id', axis=1)], axis=1) 
    return [train_raw, test_raw]

def predict(train, test):
    predictions = get_actual_y(test)
    predictions['Expected'] = train['Expected'].median()

    # train, test = cluster_data(train, test)
    # Group data by id
    train = prep_and_filter_data(train)
    test = prep_and_filter_data(test)

    # Random Forest using all data
    full_tree_train_data = train.dropna()
    full_tree_test_data = test.dropna()
    model = ensemble.RandomForestRegressor(n_estimators=N_EST, max_depth=MAX_D, n_jobs=-1, min_samples_split=MIN_LEAF, max_features=MAX_FEATURES, criterion="mae")
    full_tree_test_data['predictions'] = model.fit(X=full_tree_train_data[full_tree_train_data.columns.difference(['Id','Expected'])], y=full_tree_train_data['Expected']).predict(X=full_tree_test_data[full_tree_test_data.columns.difference(['Id','Expected'])])

    # Random Forest using only means
    partial_tree_train_data = train[train.count(1) < 45][train['Ref_mean'].notnull()][train['RhoHV_mean'].notnull()][train['Zdr_mean'].notnull()][train['Kdp_mean'].notnull()]
    partial_tree_train_data = partial_tree_train_data.loc[:,['Ref_mean','RhoHV_mean','Zdr_mean','Kdp_mean','Expected']].copy()
    partial_tree_test_data = test[test.count(1) < 45][test['Ref_mean'].notnull()][test['RhoHV_mean'].notnull()][test['Zdr_mean'].notnull()][test['Kdp_mean'].notnull()]
    partial_tree_test_data = partial_tree_test_data.loc[:,['Ref_mean','RhoHV_mean','Zdr_mean','Kdp_mean','Expected']].copy()
    partial_model = ensemble.RandomForestRegressor(n_estimators=N_EST, max_depth=MAX_D, n_jobs=-1, min_samples_split=MIN_LEAF, max_features='auto', criterion="mae")
    partial_tree_test_data['predictions'] = partial_model.fit(X=partial_tree_train_data[partial_tree_train_data.columns.difference(['Id','Expected'])], y=partial_tree_train_data['Expected']).predict(X=partial_tree_test_data[partial_tree_test_data.columns.difference(['Id','Expected'])])

    for i in partial_tree_test_data.index:
        predictions.loc[i,'Expected'] = partial_tree_test_data.loc[i,'predictions']
    predictions.loc[full_tree_test_data.index,'Expected'] = full_tree_test_data.loc[:,'predictions']
    return predictions

def run(data):
    data = data.sample(1000000)
    errors = list()
    med_errors = list()
    for t1, t2 in cross_validation.KFold(data.shape[0], n_folds=10, shuffle=True):
        # Prep data - still raw
        train = data.iloc[t1]
        test = data.iloc[t2]
        y = get_actual_y(test)
        
        e = error_rate(y['Expected'], predict(train, test)['Expected'])
        med_e = error_rate(y['Expected'], simplest_predictions(train, test)['Expected'])
        errors.append(e)
        med_errors.append(med_e)
        print("Median error rate: {} --- Error rate: {}".format(med_e, e))
        print("Difference: {}".format(med_e - e))

    print("Avg median error: {} ({})".format(statistics.mean(med_errors), statistics.stdev(med_errors)))
    print("Avg error: {} ({})".format(statistics.mean(errors), statistics.stdev(errors)))
    print("Difference in errors: {}".format(statistics.mean(med_errors) - statistics.mean(errors)))

def error_rate(expected, predicted):
    # MAE
    return (expected - predicted).abs().mean()

def prep_and_filter_data(data):
    means = data.groupby('Id').mean()
    means.columns += '_mean'
    medians = data.groupby('Id').median()
    medians.columns += '_median'
    comb = pd.concat([means, medians], axis=1)
    #comb.drop('Expected_std', axis=1, inplace=True)
    comb = comb[comb['Ref_mean'] > 0]
    comb = comb[comb['Expected_mean'] < 70]
    comb['Expected'] = comb['Expected_mean']
    comb.drop('Expected_mean', inplace=True, axis=1)
    return comb

# Data + features
# data_raw = pd.read_csv('input/train_clean.csv', usecols=[0,3,11,15,19,23])
MAX_FEATURES='auto'; N_EST=30; MAX_D=None; MIN_LEAF=1000;
run(data_raw)
 
 
train_raw = pd.read_csv('input/train_clean.csv')
test_raw = pd.read_csv('input/test.csv')
test_raw['Expected'] = test_raw['Ref'] - test_raw['Ref']
p = predict(train_raw, test_raw)
p.to_csv('output/output_{}.csv'.format(int(time.time())))
