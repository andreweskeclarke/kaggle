""" 
How Much Did it Rain? II

Kaggle competition, predict rainfall amount based on polarimetric doppler radar data.

Script based on a sample from Paul Duan.

"""

from __future__ import division

import numpy as np
from sklearn import (metrics, cross_validation, linear_model)

SEED = 42  # always use a seed for randomized procedures


def load_data(filename, use_labels=True):
    """
    Load data from CSV files and return them as numpy arrays
    The use_labels parameter indicates whether one should
    read the first column (containing class labels). If false,
    return all 0s. 
    """

    # load column 1 to 23 (24 is Y)
    data = np.genfromtxt("input/{}".format(filename),delimiter=",",
            usecols=range(0,23),skiprows=1) 
    if use_labels:
        labels = np.genfromtxt("input/{}".format(filename),delimiter=",",
                usecols=[23],skiprows=1) 
    else:
        labels = np.zeros(data.shape[0])
    return labels, data


def save_results(predictions, filename):
    """Given a vector of predictions, save results in CSV format."""
    with open(filename, 'w') as f:
        f.write("id,PREDICTION\n")
        for i, pred in enumerate(predictions):
            f.write("%d,%f\n" % (i + 1, pred))


def main():
    """
    Fit models and make predictions.
    We'll use one-hot encoding to transform our categorical features
    into binary features.
    y and X will be numpy array objects.
    """
    model = linear_model.LinearRegression()  # the classifier we'll use

    # === load data in memory === #
    print("loading data")
    y, X = load_data('small.csv')
    y_test, X_test = load_data('test.csv', use_labels=False)

    # === features === #

    # === training & metrics === #
    # Use Mean Absolute Error: http://climate.geog.udel.edu/~climate/publication_html/Pdf/WM_CR_05.pdf
    mean_mae = 0.0
    n = 10  # repeat the CV procedure 10 times to get more precise results
    for i in range(n):
        # for each iteration, randomly hold out 20% of the data as CV set
        X_train, X_cv, y_train, y_cv = cross_validation.train_test_split(
            X, y, test_size=.20, random_state=i*SEED)

        # if you want to perform feature selection / hyperparameter
        # optimization, this is where you want to do it

        # train model and make predictions
        model.fit(X_train, y_train) 
        preds = model.predict(X_cv)[:, 1]

        # compute MAE metric for this CV fold
        loss = metrics.mean_absolute_error(y_cv, preds)
        print("MAE (fold %d/%d): %f" % (i + 1, n, roc_auc))
        mean_mae += mae

    print("Mean Absolute Error: %f" % (mean_mae/n))

#    # === Predictions === #
#    # When making predictions, retrain the model on the whole training set
#    model.fit(X, y)
#    preds = model.predict(X_test)[:, 1]
#    filename = raw_input("Enter name for submission file: ")
#    save_results(preds, filename + ".csv")

if __name__ == '__main__':
    main()
