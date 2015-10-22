import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

def remove_bad_data():
    # Remove Id's that have no Ref:Kdp data over the entire hour
    df = pd.read_csv('input/train.csv')
    df['count_non_null_measures'] = df.loc[:, 'Ref':'Kdp_5x5_90th'].count(1)
    grouped = df.groupby('Id')[['count_non_null_measures','Expected','Id']].max()
    df = pd.merge(df, grouped, on='Id', how='outer')
    df['cotains_measurements_over_hour'] = df['count_non_null_measures_y'] > 0
    df = df[df['cotains_measurements_over_hour']]
    df['Expected'] = df['Expected_x']
    df.drop(['Expected_y','cotains_measurements_over_hour','Expected_x','count_non_null_measures_x','count_non_null_measures_y'], axis=1, inplace=True)
    df.loc[:10,:]

    df.to_csv('input/train_clean.csv')


##### After cleaning
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

if os.environ.get('SIZE') is not None:
    df = read_data()
    sample = df[df['Expected'] < 70].sample(int(os.environ.get('SIZE')))
    print(sample)
    sample.to_csv('input/sample.csv')
