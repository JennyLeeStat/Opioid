#
#    Copyright 2017 Jenny Lee (jennylee.stat@gmail.com)
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.
#

import os
import sys
import time
import itertools
import logging
import pickle
import numpy as np
import pandas as pd
import pyprind
import glob
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import SGDClassifier
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.linear_model import Perceptron
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import fbeta_score, accuracy_score
from sklearn.utils import shuffle

import data_prep as prep

plt.style.use('ggplot')
logging.basicConfig(
    format='%(levelname)s %(message)s',
    stream=sys.stdout, level=logging.INFO)

# ===== parameters ==================
random_state = 42
batch_dir = "dataset/batches"
test_ratio = .2
val_ratio = .15
batch_size = 256
validate_every = 50
print_every = batch_size * 2

loss = ['hinge', 'log']
alpha = [.000001, .00001, .0001, .001]
l1_ratio = [0., .1, .2, .3, .4, .5, .6, .7, .8, .9]

partial_fit_classifiers = {
    'SGD-SVM': SGDClassifier(random_state=random_state, loss='hinge'),
    'SGD-Log': SGDClassifier(random_state=random_state, loss='log'),
    'Perceptron': Perceptron(random_state=random_state),
    'NB Multinomial': MultinomialNB(alpha=0.01),
    'Passive-Aggressive': PassiveAggressiveClassifier(random_state=random_state)
}


def get_batchnames(split_val=True):
    """
    shuffle train and test set then split train set further to train set and validation set
    :return: pickle filenames of train/validation/test batches
    """
    np.random.seed(random_state)
    test_batch_names = glob.glob(os.path.join('dataset/batches/test_batches', '*.pickle'))
    test_batch_names = np.random.choice(test_batch_names, len(test_batch_names), replace=False)
    train_batch_names = glob.glob(os.path.join('dataset/batches/train_batches', '*.pickle'))
    train_batch_names = np.random.choice(train_batch_names, len(train_batch_names), replace=False)

    if split_val:
        val_batch_names = np.random.choice(
            train_batch_names, int(val_ratio * len(train_batch_names)), replace=False)
        train_batch_names = np.setdiff1d(train_batch_names, val_batch_names)
        train_batch_names = np.random.choice(train_batch_names, len(train_batch_names), replace=False)

        print("number of train batches: {}".format(len(train_batch_names)))
        print("Number of validation batches: {}".format(len(val_batch_names)))
        print("number of test batches: {}".format(len(test_batch_names)))
        return train_batch_names.tolist(), test_batch_names.tolist(), val_batch_names.tolist()
    else:
        print("number of train batches: {}".format(len(train_batch_names)))
        print("number of test batches: {}".format(len(test_batch_names)))
        return train_batch_names.tolist(), test_batch_names.tolist()


def load_batches(filename):
    with open(filename, 'rb') as f:
        features = pickle.load(f)
        label = pickle.load(f)
        f.close()
    return features, label


def batch_features_labels(features, labels, batch_size):
    for start in range(0, len(features), batch_size):
        end = min(start + batch_size, len(features))
        yield features[start:end], labels[start:end]


def load_train_batches(train_batch_names, batch_id, batch_size, random_state=random_state):
    filename = train_batch_names[batch_id]
    features, labels = load_batches(filename)
    features, labels = shuffle(features, labels, random_state=random_state)

    return batch_features_labels(features, labels, batch_size)


def load_random_test_batch(test_features, test_labels, batch_size):
    np.random.seed(random_state)
    idx = np.random.choice(len(test_features), batch_size)
    return test_features.iloc[idx, :], test_labels[idx].reshape(batch_size, 1)


def concat_batches(batch_names):
    features = None
    labels = None
    logging.info("Concatenating {} batches".format(len(batch_names)))
    pbar = pyprind.ProgBar(len(batch_names))
    for file in batch_names:
        X, y = load_batches(file)
        if features is None:
            features = X
            labels = y
        else:
            features = pd.concat([ features, X ])
            labels = np.concatenate([ labels, y ])
        pbar.update()
    return features, labels


def initialize_stats(partial_fit_classifiers):
    results = {}
    for clf_name in partial_fit_classifiers:
        stats = {
            'n_train': [],
            'train_time': [],
            'pred_time': [],
            'acc_train': [],
            'acc_test': [],
            'f_train': [],
            'f_test': []
        }
        results[ clf_name ] = stats
    return results


def train_predict(partial_fit_classifiers, train_batch_names, val_features, val_labels):
    classes = np.array([ 0, 1 ])
    results = initialize_stats(partial_fit_classifiers)
    steps = 0

    for batch_i in range(len(train_batch_names)):
        for X_train, y_train in load_train_batches(train_batch_names, batch_i, batch_size):
            for clf_name, clf in partial_fit_classifiers.items():
                steps += 1
                # train via partial fit
                tick = time.time()
                clf.partial_fit(X_train, y_train, classes=classes)
                train_time = time.time() - tick

                # predict
                tick = time.time()
                pred_train = clf.predict(X_train)
                pred_time = time.time() - tick

                pred_train = clf.predict(X_train)
                results[ clf_name ][ 'train_time' ].append(train_time)
                results[ clf_name ][ 'pred_time' ].append(pred_time)
                results[ clf_name ][ 'n_train' ].append(X_train.shape[ 0 ])
                results[ clf_name ][ 'acc_train' ].append(accuracy_score(y_train, pred_train))
                results[ clf_name ][ 'f_train' ].append(fbeta_score(y_train, pred_train, beta=.5))

        X_val, y_val = load_random_test_batch(val_features, val_labels, batch_size)
        for clf_name, clf in partial_fit_classifiers.items():
            pred_val = clf.predict(X_val)
            results[ clf_name ][ 'acc_test' ].append(accuracy_score(y_val, pred_val))
            results[ clf_name ][ 'f_test' ].append(fbeta_score(y_val, pred_val, beta=.5))

        logging.info(
            "batch {} / {} ================================".format(batch_i + 1, len(train_batch_names)))
        logging.info("Number of validation sample: {}".format(X_train.shape[ 0 ]))
        logging.info("F-beta score (beta=0.5) on training set:")
        logging.info("SGD-SVM: {}".format(results[ 'SGD-SVM' ][ 'f_train' ][ -1 ]))
        logging.info("SGD-Log: {}".format(results[ 'SGD-Log' ][ 'f_train' ][ -1 ]))
        logging.info("Perceptron: {}".format(results[ 'Perceptron' ][ 'f_train' ][ -1 ]))
        logging.info("NB Multinomial: {}".format(results[ 'NB Multinomial' ][ 'f_train' ][ -1 ]))
        logging.info("PA: {}".format(results[ 'Passive-Aggressive' ][ 'f_train' ][ -1 ]))
        # logging.info("==========================================")

    return results

def get_time_res(results):
    n_train = results['SGD-SVM']['n_train']
    res = []
    for k in partial_fit_classifiers.keys():
        train_time = results[k]['train_time']
        pred_time = results[k]['pred_time']
        mean_train_time = np.mean([10000 * t / n_train[i] for i, t in enumerate(train_time)])
        mean_pred_time = np.mean([10000 * t / n_train[i] for i, t in enumerate(pred_time)])
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
    plt.ylabel('runtime (sec)')
    plt.title('Runtime per 10,000 instances')

    if not os.path.isdir("assets"):
        os.mkdir("assets")
    plt.savefig("assets/time_res.png")

    plt.show()


def plot_score(results, window=20):
    my_col = sns.color_palette("husl", 5)
    plt.figure(figsize=(11, 11))
    ax1 = plt.subplot(221)

    for i, clf_name in enumerate(partial_fit_classifiers.keys()):
        plt.plot(pd.Series(results[ clf_name ][ 'acc_train' ]).rolling(window=window).mean(),
                 color=my_col[ i ], linewidth=2.5)
    plt.ylabel("Accuracy")
    plt.xlabel("Training steps")
    plt.legend(loc="best", labels=partial_fit_classifiers.keys())
    plt.title("Accuracy on training set")

    ax2 = plt.subplot(222, sharey=ax1)
    for i, clf_name in enumerate(partial_fit_classifiers.keys()):
        plt.plot(pd.Series(results[ clf_name ][ 'f_train' ]).rolling(window=window).mean(),
                 color=my_col[ i ], linewidth=2.5)
    plt.ylabel("F-score")
    plt.xlabel("Training steps")
    # plt.legend(loc="best", labels=partial_fit_classifiers.keys())
    plt.title("F score (beta=0.5) on training set")
    plt.savefig("assets/compare_score.png")
    plt.show()


def get_test_score(results):
    acc_test = {}
    f_test = {}
    for clf_names in partial_fit_classifiers.keys():
        acc_test[clf_names] = results[clf_names]['acc_test']
        f_test[clf_names] = results[clf_names]['f_test']
    acc_test = pd.DataFrame(acc_test)
    f_test = pd.DataFrame(f_test)
    f_test_mean = pd.DataFrame(f_test.mean())
    f_test_mean.columns = ['f_test_mean']
    acc_test_mean = pd.DataFrame(acc_test.mean())
    acc_test_mean.columns = ['acc_test_mean']
    mean_scores = acc_test_mean.join(f_test_mean).sort_values(by='f_test_mean', ascending=False)
    return mean_scores, acc_test, f_test


def eval_error(clf, val_features, val_labels, n_to_try=100):
    val_fscore = [ ]
    val_accuracy = [ ]
    try:
        for _ in range(n_to_try):
            X_val, y_val = load_random_test_batch(val_features, val_labels, batch_size)
            pred_val = clf.predict(X_val)
            val_fscore.append(fbeta_score(y_val, pred_val, beta=.5))
            val_accuracy.append(accuracy_score(y_val, pred_val))

    except StopIteration:
        return None

    return np.mean(val_fscore), np.mean(val_accuracy)


def sgd_grid_search(train_batch_names, val_features, val_labels,
                    loss, alpha, l1_ratio, random_state, save_res=True):
    classes = np.array([ 0, 1 ])
    params = list(itertools.product(alpha, l1_ratio))
    params_ = list(itertools.product(loss, params))
    param_dict = [ {'loss': c[ 0 ],
                    'alpha': c[ 1 ][ 0 ],
                    'l1_ratio': c[ 1 ][ 1 ],
                    'val_f_score': 0.} for c in params_ ]

    start = time.time()
    for i, p in enumerate(param_dict):
        clf = SGDClassifier(loss=p[ 'loss' ], alpha=p[ 'alpha' ], l1_ratio=p[ 'l1_ratio' ],
                            penalty='elasticnet', random_state=random_state)
        print('')
        logging.info("{}/{} Evaluating hyperparameters: ".format(i + 1, len(param_dict)))
        logging.info("loss: {}, alpha: {}, l1_ratio: {}".format(p[ 'loss' ], p[ 'alpha' ], p[ 'l1_ratio' ]))

        for batch_i in range(len(train_batch_names)):
            for X_train, y_train in load_train_batches(train_batch_names, batch_i, batch_size):
                clf.partial_fit(X_train, y_train, classes=classes)

        val_fscore, val_accuracy = eval_error(clf, val_features, val_labels)
        p[ 'val_f_score' ] = val_fscore
        logging.info('Validation F-score: {}'.format(val_fscore))

    logging.info("Total time elapsed: {} seconds".format(time.time() - start))

    res = pd.DataFrame(param_dict)
    if save_res:
        pickle_name = 'results/grid_search_res.pickle'
        with open(pickle_name, 'wb') as f:
            pickle.dump(res, f)
        f.close()
        logging.info("grid seach result is saved as: {}".format(pickle_name))

    return res

    res = pd.DataFrame(param_dict)
    if save_res:
        pickle_name = 'grid_search.pickle'
        with open(pickle_name, 'wb') as f:
            pickle.dump(res, f)
        f.close()
        logging.info("grid seach result is saved as: {}".format(pickle_name))

    return res

def main():
    npi = prep.prepare_npi(prep.npi_url)

    # split data batches
    train_batch_names, test_batch_names, val_batch_names = get_batchnames(split_val=True)
    val_features, val_labels = concat_batches(val_batch_names)
    test_features, test_labels = concat_batches(test_batch_names)

    # select the best model
    results = train_predict(partial_fit_classifiers, train_batch_names, val_features, val_labels)
    mean_scores, _, _ = get_test_score(results)
    logging.info("Avg test scores: {}".format(mean_scores))

    search_res = sgd_grid_search(train_batch_names, val_features, val_labels, loss, alpha, l1_ratio, random_state)
    best_param = search_res.sort_values(by='val_f_score', ascending=False).iloc[ 0 ]
    logging.info("Best parameters selected: {}".format(best_param))

    # retrain the best model with all training samples
    train_batch_names, _ = get_batchnames(split_val=False)
    best_clf = SGDClassifier(loss=best_param[ 'loss' ],
                             alpha=best_param[ 'alpha' ],
                             l1_ratio=best_param[ 'l1_ratio' ],
                             random_state=random_state,
                             average=True)
    classes = np.array([ 0, 1 ])

    pbar = pyprind.ProgBar(len(train_batch_names))
    for batch_i in range(len(train_batch_names)):
        for X_train, y_train in load_train_batches(train_batch_names, batch_i, batch_size):
            best_clf.partial_fit(X_train, y_train, classes=classes)
        pbar.update()

    final_f_score, final_accuracy = eval_error(best_clf, test_features, test_labels)
    logging.info("Test F-score: {}".format(final_f_score))
    logging.info("Test accuracy: {}".format(final_accuracy))

    filename = 'results/best_model.pickle'
    with open(filename, 'wb') as f:
        pickle.dump(best_clf, f)
    logging.info("Best model is saved as {}".format(filename))



if __name__ == "__main__":
     main()

#































