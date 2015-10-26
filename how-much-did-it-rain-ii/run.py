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
from sklearn import cross_validation
from sklearn.metrics import mean_absolute_error

#plt.scatter(x=train_pca[:,0], y=train_pca[:,1], c=cluster.k_means(train_pca, cluster_size)[1])
#plt.show()

def read_data():
    df = pd.read_csv('input/train_clean.csv')
    df['count_non_null_measures'] = df.loc[:, 'Ref':'Kdp_5x5_90th'].count(1)
    grouped = df.groupby('Id')[['count_non_null_measures','Expected','Id']].max()
    df = pd.merge(df, grouped, on='Id', how='outer')
    df['cotains_measurements_over_hour'] = df['count_non_null_measures_x'] == 20
    df = df[df['cotains_measurements_over_hour']]
    df['Expected'] = df['Expected_x']
    df.drop(['Expected_y','cotains_measurements_over_hour','Expected_x','count_non_null_measures_x','count_non_null_measures_y','Unnamed: 0'], axis=1, inplace=True)
    return df

def build_models(train, cluster_size):
    # Normalize before building PCA components
    train_norm = preprocessing.scale(train.loc[:,'minutes_past':'Kdp_5x5_90th'])
    pca = decomposition.PCA(n_components=9).fit(train_norm)
    train_pca = pca.transform(train_norm)
    # Cluster measurements based on PCA components
    train_kmeans = cluster.KMeans(n_clusters=cluster_size, n_init=25, max_iter=1000, init='k-means++').fit(train_pca)
    train_categories = train_kmeans.predict(train_pca)
    train_dummies = pd.get_dummies(train_categories)
    col_names = []
    for i in range(0,cluster_size):
        col_names.append('cat' + str(i))
    train_dummies.columns = col_names
    train_dummies.set_index(train.index, inplace=True)
    train_dummies['Id'] = train['Id']
    train = pd.merge(left=train, right=train_dummies)

    # Build simple linear model
    model = linear_model.LinearRegression()
    model.fit(X=train[train.columns - ['Id','Expected']], y=train['Expected'])
    
    return [pca, train_kmeans, model]

def build_predictions(test, pca, kmeans, model, cluster_size):
    test_norm = preprocessing.scale(test.loc[:,'minutes_past':'Kdp_5x5_90th'])
    test_pca = pca.transform(test_norm)
    test_dummies = pd.get_dummies(kmeans.predict(test_pca))
    col_names = []
    for i in range(0,cluster_size):
        col_names.append('cat' + str(i))
    test_dummies.columns = col_names
    test_dummies.set_index(test.index, inplace=True)
    test_dummies['Id'] = test['Id']
    test = pd.concat([test, test_dummies], axis=1)
    return model.predict(X=test[test.columns - ['Id','Expected']])
    
def run_cv(sample, cluster_size, n_kfolds):
    kf = cross_validation.KFold(sample.shape[0], n_folds=n_kfolds, shuffle=True)
    mae_sum = 0
    count = 0
    for t1, t2 in kf:
        count += 1
        print("CV run {} out of {}...".format(count, n_kfolds))
        train = sample.iloc[t1]
        test = sample.iloc[t2]
        pca, kmeans, model = build_models(train, cluster_size)
        test['predict'] = build_predictions(test, pca, kmeans, model, cluster_size)
        mae_sum = mae_sum + mean_absolute_error(test['Expected'], test['predict'])

    mae_avg = mae_sum / n_kfolds
    print("Expected MAE is: {}".format(mae_avg))

def average_weighted_by_minutes(group):
    count = group.shape[0]
    expected_w_avg = 0
    prev_minute = 0
    for i in range(0,count):
        row = group.iloc[i,:]
        multiplier = row['minutes_past'] - prev_minute
        prev_minute = row['minutes_past']
        expected_w_avg += (multiplier * group.iloc[i,:]['Expected'])/60
    multiplier = 60 - prev_minute
    expected_w_avg += (multiplier * group.iloc[count-1,:]['Expected'])/60
    return pd.Series([group.iloc[0,:]['Id'], expected_w_avg ], index = ['Id', 'Expected'])

if __name__ == "__main__":
    sample_size = 100000
    cluster_size = 7
    n_kfolds = 10
    #if os.environ.get('FRESH_SAMPLE') is not None:
    #    df = read_data()
    #    sample = df[df['Expected'] < 70].sample(sample_size)
    #else:
    #    sample = pd.read_csv('input/sample.csv')

    print("Read data...")
    df = read_data()
    print("Build models...")
    pca_tot, kmeans_tot, model_tot = build_models(df[df['Expected'] < 70], cluster_size)
    print("Read test data...")
    test_tot = pd.read_csv('input/test.csv')
    test_tot.fillna(0, inplace=True) # Hack

    test_tot['Expected'] = test_tot['Id'] - test_tot['Id'] # Hack
    print("Build predictions...")
    test_tot['Expected'] = build_predictions(test_tot, pca_tot, kmeans_tot, model_tot, cluster_size)

    print("Summarize output...")
    output = test_tot.groupby('Id').apply(average_weighted_by_minutes)
    print("Out to csv...")
    csv_name = 'output/output_{}.csv'.format(int(time.time()))
    output.to_csv(csv_name, columns=['Expected'])
