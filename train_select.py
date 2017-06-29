import os
import sys
import time
import logging
import pickle
import numpy as np
import pandas as pd
import pyprind
from glob import glob
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import SGDClassifier
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.linear_model import Perceptron
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import fbeta_score, accuracy_score

logging.basicConfig(
    format='%(levelname)s %(message)s',
    stream=sys.stdout, level=logging.INFO)

random_state = 42
batch_dir = "dataset/batches"
test_ratio = .15
validate_every = 10
print_every = 10


partial_fit_classifiers = {
    'SGD': SGDClassifier(random_state=random_state, loss='log'),
    'Perceptron': Perceptron(random_state=random_state),
    'NB Multinomial': MultinomialNB(alpha=0.01),
    'Passive-Aggressive': PassiveAggressiveClassifier(random_state=random_state)
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
    np.random.RandomState(random_state)
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
            'n_train': [],
            'n_train_pos': [],
            'train_time': [],
            'pred_time': [],
            'acc_train': [],
            'acc_test': [],
            'f_train': [],
            'f_test': []
        }
        results[ clf_name ] = stats
    return results


def train_predict(partial_fit_classifiers, train_batch_names, n_train):
    classes = np.array([ 0, 1 ])
    files_to_read = train_batch_names[ :n_train ]
    results = initialize_stats(partial_fit_classifiers)
    # pbar = pyprind.ProgBar(n_train)
    for i, filename in enumerate(files_to_read):
        X_train, y_train = load_batches(filename)

        for clf_name, clf in partial_fit_classifiers.items():
            # train via partial fit
            tick = time.time()
            clf.partial_fit(X_train, y_train, classes=classes)
            train_time = time.time() - tick

            # predict
            tick = time.time()
            pred_train = clf.predict(X_train)
            pred_time = time.time() - tick

            results[ clf_name ][ 'train_time' ].append(train_time)
            results[ clf_name ][ 'pred_time' ].append(pred_time)
            results[ clf_name ][ 'n_train' ].append(X_train.shape[ 0 ])
            results[ clf_name ][ 'n_train_pos' ].append(sum(y_train))
            results[ clf_name ][ 'acc_train' ].append(accuracy_score(y_train, pred_train))
            results[ clf_name ][ 'f_train' ].append(fbeta_score(y_train, pred_train, beta=.5))

        if i % validate_every == 0 and i > 0:
            X_val, y_val = X_train, y_train
            pred_val = clf.predict(X_val)
            results[ clf_name ][ 'acc_test' ].append(accuracy_score(y_val, pred_val))
            results[ clf_name ][ 'f_test' ].append(fbeta_score(y_val, pred_val, beta=.5))

            # After validation, we'll use the val batch to  update our model
            clf.partial_fit(X_train, y_train, classes=classes)
            pred_train = clf.predict(X_train)
            results[ clf_name ][ 'n_train' ].append(X_train.shape[ 0 ])
            results[ clf_name ][ 'n_train_pos' ].append(sum(y_train))
            results[ clf_name ][ 'acc_train' ].append(accuracy_score(y_train, pred_train))
            results[ clf_name ][ 'f_train' ].append(fbeta_score(y_train, pred_train, beta=.5))

        if i % print_every == 0 and i > 0:
            logging.info("iter {} / {}".format(i, n_train))
            logging.info("Number of validation sample: {}".format(X_train.shape[ 0 ]))
            logging.info("===== F-beta score (beta=0.5) ==========")
            logging.info("SGD: {}".format(results[ 'SGD' ][ 'f_train' ][ -1 ]))
            logging.info("Perceptron: {}".format(results[ 'Perceptron' ][ 'f_train' ][ -1 ]))
            logging.info("NB Multinomial: {}".format(results[ 'NB Multinomial' ][ 'f_train' ][ -1 ]))
            logging.info("P-A: {}".format(results[ 'Passive-Aggressive' ][ 'f_train' ][ -1 ]))
            logging.info("========================================")

            # pbar.update()

    return results

def get_time_res(results):
    n_train = results['SGD']['n_train']
    res = []
    for k in partial_fit_classifiers.keys():
        train_time = results[k]['train_time']
        pred_time = results[k]['train_time']
        mean_train_time = np.mean([5000 * t / n_train[i] for i, t in enumerate(train_time)])
        mean_pred_time = np.mean([5000 * t / n_train[i] for i, t in enumerate(pred_time)])
        res.append((k, mean_train_time, mean_pred_time))
    return res


def plot_time(res):
    res = pd.DataFrame(res)
    res.columns = [ 'clf_name', 'training', 'prediction' ]
    res[ 'total' ] = res[ 'training' ] + res[ 'prediction' ]
    res = res.sort_values(by='total')
    res_melt = pd.melt(res, id_vars=[ 'clf_name' ], value_vars=[ 'prediction', 'training', 'total' ])
    res_melt.columns = [ 'clf_name', 'time', 'value' ]

    plt.figure(figsize=(11, 5))
    sns.barplot(x="clf_name", y="value", hue="time", data=res_melt,
                palette=sns.color_palette("husl", 3),
                linewidth=1, edgecolor=".2")
    plt.xlabel('')
    plt.ylabel('runtime(sec)')
    plt.title('Runtime per 5,000 instances')
    plt.savefig("assets/time_res.png")
    plt.show()


def plot_score(results):
    my_col = sns.color_palette("husl", 4)
    plt.figure(figsize=(11, 5))
    ax1 = plt.subplot(121)
    n = np.cumsum(results[ 'SGD' ][ 'n_train' ])
    len_n = len(n)
    for i, clf_name in enumerate(partial_fit_classifiers.keys()):
        plt.plot(n, pd.Series(results[ clf_name ][ 'acc_train' ][ :len_n ]).rolling(window=5).mean(),
                 color=my_col[ i ], linewidth=2.5)
    plt.ylabel("Accuracy")
    plt.xlabel("Training samples (#)")
    plt.legend(loc="best", labels=partial_fit_classifiers.keys())
    plt.title("Accuracy on training set")

    ax2 = plt.subplot(122, sharey=ax1)
    for i, clf_name in enumerate(partial_fit_classifiers.keys()):
        plt.plot(n, pd.Series(results[ clf_name ][ 'f_train' ][ :len_n ]).rolling(window=5).mean(),
                 color=my_col[ i ], linewidth=2.5)
    plt.ylabel("F-score")
    plt.xlabel("Training samples (#)")
    plt.title("F-score (beta=0.5) on training set")
    plt.savefig("assets/compare_score.png")
    plt.show()


def main():
    test_batch_names, train_batch_names = split_test(batch_dir=batch_dir, test_ratio=test_ratio)
    n_train = len(train_batch_names)
    results = train_predict(partial_fit_classifiers, train_batch_names, n_train=n_train)

    filename = "results.pickle"
    with open(filename, 'wb') as f:
        pickle.dump(results, f)
        f.close()
    return


if __name__ == "__main__":
    main()
