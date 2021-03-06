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
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import pyprind

from sklearn.metrics import fbeta_score
import tensorflow as tf
from tensorflow.contrib import slim
import train_select as ts

warnings.filterwarnings('ignore')


# parameters ======================
random_state = 42
n_classes = 1
n_features = 535
n_hidden1 = 64
n_hidden2 = 32
keep_prob = .7

batch_size = 256
learning_rate = .0001
n_epochs = 25
val_every = 50
write_every = 1
print_every = 500

def load_random_test_batch(test_features, test_labels, batch_size):
    idx = np.random.choice(len(test_features), batch_size)
    return test_features.iloc[idx, :], labels_to_2d(test_labels[idx])


def batch_features_labels(features, labels, batch_size):
    for start in range(0, len(features), batch_size):
        end = min(start + batch_size, len(features))
        yield features[start:end], labels_to_2d(labels[start:end])


def load_train_batches(train_batch_names, batch_id, batch_size):
    filename = train_batch_names[batch_id]
    features, labels = ts.load_batches(filename)
    return batch_features_labels(features, labels, batch_size)

def labels_to_2d(labels):
    tmp = np.zeros((len(labels), 2), np.int32)
    tmp[:, 1] = labels
    tmp[tmp[:, 1]==0, 0] = 1
    return tmp

def f_score(recall, precision, beta=0.5):
    f = (1 + beta ** 2) * precision * recall /(beta ** precision + recall)
    return f


def dnn_model(inputs, n_hidden1, n_hidden2, n_classes, is_training=True, scope='dnn'):
    with tf.variable_scope(scope, 'dnn', [ inputs ]):
        with slim.arg_scope([ slim.fully_connected ],
                            activation_fn=tf.nn.relu,
                            weights_regularizer=slim.l2_regularizer(.15)):
            # Layer 1
            net = slim.fully_connected(inputs, n_hidden1, scope='fc1')
            net = slim.dropout(net, keep_prob, is_training=is_training)

            # Layer 2
            net = slim.fully_connected(net, n_hidden2, scope='fc2')
            net = slim.dropout(net, keep_prob, is_training=is_training)

            # output layer
            logits = slim.fully_connected(net, n_classes, activation_fn=None, scope='prediction')

            return logits


def train():
    train_batch_names, test_batch_names, val_batch_names = ts.get_batchnames(split_val=True)
    test_features, test_labels = ts.concat_batches(val_batch_names)

    with tf.Graph().as_default():
        tf.logging.set_verbosity(tf.logging.INFO)

        # inputs
        X = tf.placeholder(tf.float32, (None, n_features), 'input_features')
        y = tf.placeholder(tf.int32, (None), 'labels')

        # dnn model
        logits = dnn_model(X, n_hidden1, n_hidden2, n_classes, is_training=True)
        logits = tf.identity(logits, name='logits')

        # loss and optimizer
        preds = tf.cast(tf.round(tf.sigmoid(logits)), tf.int32)
        loss = tf.losses.mean_squared_error(labels=tf.cast(y, tf.float32), predictions=tf.sigmoid(logits))
        optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
        train_op = optimizer.minimize(loss)

        # evaluation metrics
        accuracy = tf.contrib.metrics.accuracy(labels=y, predictions=preds)
        #recall, recall_update_op = tf.contrib.metrics.streaming_recall(labels=y, predictions=preds)
        #precision, precision_update_op = tf.contrib.metrics.streaming_precision(labels=y, predictions=preds)

        stats = {
            'steps': [ ],
            'loss': [ ],
            'acc_train': [ ],
            'acc_val': [ ],
            'f_train': [ ],
            'f_val': [ ]
        }
        steps = 0
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            sess.run(tf.local_variables_initializer())

            for e in range(n_epochs):
                for batch_i in range(len(train_batch_names)):
                    for batch_features, batch_labels in load_train_batches(train_batch_names, batch_i, batch_size):
                        steps += 1
                        feed = {X: batch_features, y: batch_labels}
                        _ = sess.run(train_op, feed_dict=feed)
                        loss_, batch_preds, batch_acc = sess.run([ loss, preds, accuracy ], feed_dict=feed)

                        if steps % val_every == 0:
                            val_X, val_y = load_random_test_batch(test_features, test_labels, batch_size)
                            val_feed = {X: val_X, y: val_y}

                            val_preds, val_acc = sess.run([ preds, accuracy ], feed_dict=val_feed)
                            val_fscore = fbeta_score(y_true=val_y, y_pred=val_preds, beta=0.5)
                            batch_fscore = fbeta_score(y_true=batch_labels, y_pred=batch_preds, beta=0.5)

                        if steps % write_every == 0:
                            stats[ 'steps' ].append(steps)
                            stats[ 'loss' ].append(loss_)
                            stats[ 'acc_train' ].append(batch_acc)
                            stats[ 'acc_val' ].append(val_acc)
                            stats[ 'f_train' ].append(batch_fscore)
                            stats[ 'f_val' ].append(val_fscore)

                        if steps % print_every == 0:
                            print("Epoch {}/{} Batch {}/{} -------------------".format(e + 1, n_epochs, batch_i,
                                                                                       len(train_batch_names)))
                            print("Train accuracy: {:.4f}".format(batch_acc))
                            print("Train F-score(beta=0.5): {:.4f}".format(batch_fscore))
                            print("Val accuracy: {:4f}".format(val_acc))
                            print("Val F-score(beta=0.5): {:.4f}".format(val_fscore))
                            print("")

    return stats


def plot_score(stats, windows=100):
    res = pd.DataFrame(stats)
    res = res.loc[ :, [ 'steps', 'loss', 'acc_train', 'acc_val', 'f_train', 'f_val' ] ]
    my_col = sns.color_palette("husl", 3)
    plt.figure(figsize=(12, 6))
    ax1 = plt.subplot(131)
    plt.plot(res[ 'steps' ], res[ 'acc_train' ].rolling(window=windows).mean(),
             color=my_col[ 0 ], linewidth=2.5)
    plt.plot(res[ 'steps' ], res[ 'acc_val' ].rolling(window=windows).mean(),
             color=my_col[ 1 ], linewidth=2.5)
    plt.ylabel("Accuracy")
    plt.xlabel("Training steps")
    plt.legend(loc="best")
    plt.title("Accuracy")

    ax2 = plt.subplot(132, sharey=ax1)
    plt.plot(res[ 'steps' ], res[ 'f_train' ].rolling(window=windows).mean(),
             color=my_col[ 0 ], linewidth=2.5)
    plt.plot(res[ 'steps' ], res[ 'f_val' ].rolling(window=windows).mean(),
             color=my_col[ 1 ], linewidth=2.5)
    plt.ylabel("F-score")
    plt.xlabel("Training steps")
    plt.legend(loc="best")
    plt.title("F score (beta=0.5)")

    ax3 = plt.subplot(133)
    plt.plot(res[ 'steps' ], res[ 'loss' ].rolling(window=windows).mean(), color=my_col[ 2 ])
    plt.xlabel("Training steps")
    plt.title("Loss")
    # plt.savefig("assets/nn_compare_score.png")
    plt.show()


def test_model(saved_model_path, test_batch_names):
    loaded_graph = tf.Graph()
    with tf.Session(graph=loaded_graph) as sess:
        # saver.restore(sess, saved_model_path)
        loader = tf.train.import_meta_graph(saved_model_path + '.meta')
        loader.restore(sess, saved_model_path)
        loaded_X = loaded_graph.get_tensor_by_name('X:0')
        loaded_y = loaded_graph.get_tensor_by_name('y:0')
        loaded_keep_prob = loaded_graph.get_tensor_by_name('keep_prob:0')
        loaded_preds = loaded_graph.get_tensor_by_name('loss/preds:0')
        loaded_acc = loaded_graph.get_tensor_by_name('eval/accuracy_1:0')

        steps = 0
        stats = {
            'steps': [ ],
            'acc_test': [ ],
            'f_test': [ ]
        }
        pbar = pyprind.ProgBar(len(test_batch_names))
        for batch_i in range(len(test_batch_names)):
            for batch_features, batch_labels in load_train_batches(test_batch_names, batch_i, batch_size):
                steps += 1
                feed = {loaded_X: batch_features, loaded_y: batch_labels, loaded_keep_prob: 1.}
                preds, test_acc = sess.run([ loaded_preds, loaded_acc ], feed_dict=feed)
                test_fscore = fbeta_score(y_true=np.argmax(batch_labels, 1), y_pred=preds, beta=0.5)
                stats[ 'steps' ].append(steps)
                stats[ 'acc_test' ].append(test_acc)
                stats[ 'f_test' ].append(test_fscore)
            pbar.update()

    return stats



if __name__ == '__main__':
    stats = train()
    plot_score(stats)
