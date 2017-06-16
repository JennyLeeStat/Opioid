import sys
import logging
import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from scipy.stats import boxcox


import utils


logging.basicConfig(
    format= '%(levelname)s %(message)s',
    stream=sys.stdout, level=logging.INFO)


def load_dataset():
    opioids = pd.read_csv("dataset/opioids.csv")
    logging.info("Dataset opioids is loaded. Shape: {}".format(opioids.shape))

    overdose = pd.read_csv("dataset/overdoses.csv", thousands=',')
    logging.info("Dataset overdose is loaded. Shape: {}".format(overdose.shape))

    prescriber = pd.read_csv("dataset/prescriber-info.csv")
    logging.info("Dataset prescriber is loaded. Shape: {}".format(prescriber.shape))

    return opioids, overdose, prescriber


def num_cat_names(prescriber):
    credentials_vars = utils.clean_creds(prescriber[ 'Credentials' ])
    logging.info("Credential column label is binarized")

    cat_attribs = [ 'Gender', 'State', 'Specialty' ]
    y = [ 'Opioid.Prescriber' ]
    not_used = [ 'NPI', 'Credentials' ]
    num_attribs = np.setdiff1d(prescriber.columns, cat_attribs + y + not_used)
    nonop_names = np.setdiff1d(num_attribs, utils.opioids_columns)
    op_names = num_attribs[ [ c in utils.opioids_columns for _, c in enumerate(num_attribs) ] ]
    return y, op_names, nonop_names, cat_attribs, credentials_vars


def prepare_cat(prescriber, cat_attribs):
    return pd.get_dummies(prescriber.ix[:, cat_attribs])


def boxcox_transform(df):
    transformed = df.copy()
    param = []
    for i in range(df.shape[1]):
        #print(transformed.columns[i])
        col = transformed.ix[:, i].astype(float) + .01
        transformed.ix[:, i], p = boxcox(col)
        param.append(p)
        sys.stdout.write("\r>> Progress:" + \
                         str(i + 1) + "/" + str(df.shape[1]))
        sys.stdout.flush()
    return transformed, param


def prepare_num(prescriber, op_names, nonop_names, add_total=True):
    op = prescriber.ix[ :, op_names ]
    nonop = prescriber.ix[ :, nonop_names ]

    if add_total:
        op[ 'op_total' ] = op.sum(axis=1)
        nonop[ 'nonop_total' ] = nonop.sum(axis=1)

    num = pd.concat([ op, nonop ], axis=1)
    logging.info("Box-Cox transformation")
    transformed, param = boxcox_transform(num)

    logging.info("Min-Max scaling")
    scaler = MinMaxScaler()
    transformed = scaler.fit_transform(transformed)
    transformed = pd.DataFrame(data=transformed, index=num.index, columns=num.columns)

    return op, nonop, transformed, param


def split_data(labels, features, dest_path="dataset/datasets.pickle"):
    X_train, X_test, y_train, y_test = train_test_split(
        features, labels, test_size=0.15, random_state=42)
    logging.info("Training set has {} samples.".format(X_train.shape[ 0 ]))
    logging.info("Testing set has {} samples.".format(X_test.shape[ 0 ]))

    with open(dest_path, "wb") as f:
        pickle.dump(X_train, f)
        pickle.dump(X_test, f)
        pickle.dump(y_train, f)
        pickle.dump(y_test, f)
        f.close()
    logging.info("train and test pickles are saved in" + "dest_path")



def main():
    opioids, overdose, prescriber = load_dataset()
    y, op_names, nonop_names, cat_attribs, credentials_vars = num_cat_names(prescriber)
    cat_prepared = prepare_cat(prescriber, cat_attribs)
    op, nonop, transformed, param = prepare_num(prescriber, op_names, nonop_names, add_total=True)
    labels = pd.concat([prescriber['Opioid.Prescriber'], transformed[op_names]], axis=1)
    features = pd.concat([cat_prepared, transformed[nonop_names]], axis=1)
    split_data(labels, features)




if __name__ == "__main__":
    main()



