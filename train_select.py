import os
import sys
import time
import logging
import pickle
import pandas as pd
import numpy as np
import pyprind
from glob import glob

from sklearn.linear_model import SGDClassifier
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.linear_model import Perceptron
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import fbeta_score, accuracy_score

logging.basicConfig(
    format='%(levelname)s %(message)s',
    stream=sys.stdout, level=logging.INFO)

np.random.RandomState(42)
batch_dir = "dataset/batches"
test_ratio = .2
validate_every = 10
print_every = 5

partial_fit_classifiers = {
    'SGD': SGDClassifier(),
    'Perceptron': Perceptron(),
    'NB Multinomial': MultinomialNB(alpha=0.01),
    'Passive-Aggressive': PassiveAggressiveClassifier()
}


def load_batches(filename):
    with open(filename, 'rb') as f:
        features = pickle.load(f)
        label = pickle.load(f)
        f.close()
    return features, label


def split_test(batch_dir=batch_dir, test_ratio=test_ratio):
    filepath_list = glob(os.path.join(batch_dir, "*.pickle"))
    n_batches = len(filepath_list)
    n_test = int(test_ratio * n_batches)
    test_batch_names = np.random.choice(filepath_list, n_test, replace=False)
    train_batch_names = np.setdiff1d(filepath_list, test_batch_names)
    logging.info("Number of test batches: {}".format(len(test_batch_names)))
    logging.info("Number of train batches: {}".format(len(train_batch_names)))
    return test_batch_names, train_batch_names


def initialize_stats(partial_fit_classifiers):
    results = {}
    for clf_name in partial_fit_classifiers:
        stats = {
            't0': time.time(),
            'n_train': 0,
            'n_train_pos': 0,
            'train_time': 0.,
            'pred_time': 0.,
            'acc_train': 0.,
            'acc_test': 0.,
            'f_train': 0.,
            'f_test': 0.
        }
        results[ clf_name ] = stats
    return results



def train_predict(partial_fit_classifiers, train_batch_names, n_train):
    classes = np.array([ 0, 1 ])
    files_to_read = train_batch_names[ :n_train ]
    results = initialize_stats(partial_fit_classifiers)
    pbar = pyprind.ProgBar(n_train)
    for i, filename in enumerate(files_to_read):
        X_train, y_train = load_batches(filename)

        for clf_name, clf in partial_fit_classifiers.items():
            start = time.time()

            clf.partial_fit(X_train, y_train, classes=classes)
            pred_train = clf.predict(X_train)
            results[ clf_name ][ 'n_train' ] += X_train.shape[ 0 ]
            results[ clf_name ][ 'n_train_pos' ] += sum(y_train)
            results[ clf_name ][ 'train_time' ] = time.time() - start
            results[ clf_name ][ 'acc_train' ] = accuracy_score(y_train, pred_train)
            results[ clf_name ][ 'f_train' ] = fbeta_score(y_train, pred_train, beta=.5)

            if i % validate_every == 0 and i > 0:
                X_val, y_val = X_train, y_train
                pred_val = clf.predict(X_val)
                results[ clf_name ][ 'acc_test' ] = accuracy_score(y_val, pred_val)
                results[ clf_name ][ 'f_test' ] = fbeta_score(y_val, pred_val, beta=.5)

                # After validation, we'll use this set to  update our model
                clf.partial_fit(X_train, y_train, classes=classes)
                pred_train = clf.predict(X_train)
                results[ clf_name ][ 'n_train' ] += X_train.shape[ 0 ]
                results[ clf_name ][ 'n_train_pos' ] += sum(y_train)
                results[ clf_name ][ 'train_time' ] = time.time() - start
                results[ clf_name ][ 'acc_train' ] = accuracy_score(y_train, pred_train)
                results[ clf_name ][ 'f_train' ] = fbeta_score(y_train, pred_train, beta=.5)

        if i % print_every == 0 and i > 0:
            logging.info("iter {} / {}".format(i, n_train))
            logging.info("Number of training sample: {}".format(X_train.shape))
            logging.info("F-beta score (beta=0.5)")
            logging.info("SGD: {}".format(results['SGD']['f_train'][-1]))
            logging.info("Perceptron: {}".format(results[ 'Perceptron' ][ 'f_train' ][ -1 ]))
            logging.info("NB Multinomial: {}".format(results[ 'NB Multinomial' ][ 'f_train' ][ -1 ]))
            logging.info("P-A: {}".format(results['Passive-Aggressive']['f_train'][-1]))

        pbar.update()

    return results

def main():
    test_batch_names, train_batch_names = split_test(batch_dir=batch_dir, test_ratio=test_ratio)
    results = train_predict(partial_fit_classifiers, train_batch_names, n_train = 10)
    return results

if __name__ == "__main__":
    main()
