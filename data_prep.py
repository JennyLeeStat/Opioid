import sys
import pickle
import numpy as np
import pandas as pd
import logging

logging.basicConfig(
    format='%(levelname)s %(message)s',
    stream=sys.stdout, level=logging.INFO)

import warnings
warnings.filterwarnings('ignore')

import utils

drugs_url = "http://download.cms.gov/Research-Statistics-Data-and-Systems/Statistics-Trends-and-Reports/Medicare-Provider-Charge-Data/Downloads/PartD_Prescriber_PUF_NPI_DRUG_15.zip"
npi_url = "http://download.cms.gov/Research-Statistics-Data-and-Systems/Statistics-Trends-and-Reports/Medicare-Provider-Charge-Data/Downloads/PartD_Prescriber_PUF_NPI_15.zip"
st_url = "http://download.cms.gov/Research-Statistics-Data-and-Systems/Statistics-Trends-and-Reports/Medicare-Provider-Charge-Data/Downloads/PartD_Prescriber_PUF_Drug_St_15.zip"
ntl_url = "http://download.cms.gov/Research-Statistics-Data-and-Systems/Statistics-Trends-and-Reports/Medicare-Provider-Charge-Data/Downloads/PartD_Prescriber_PUF_Drug_Ntl_15.zip"

dest_dir = "dataset"
chunk_size = 100000
n_other_drugs = 50
n_batches_to_try = 10000
random_state = 42


def prepare_npi(npi_url, dropna=True, add_new_features=True):
    utils.download_and_decompress(npi_url, dest_dir)
    filename = dest_dir + "/" + npi_url.split("/")[ -1 ].split(".")[ 0 ] + ".txt"
    npi = pd.read_table(filename)
    # TODO: add verbose flag, and print out preprocessing progress

    # ===== Feature selection =====
    cols_to_keep = [ 'npi', 'nppes_provider_gender', 'nppes_provider_zip5',
                     'nppes_provider_state', 'nppes_provider_country',
                     'specialty_description', 'medicare_prvdr_enroll_status',
                     'total_day_supply', 'bene_count',
                     'beneficiary_average_risk_score',
                     'antipsych_claim_count_ge65',
                     'hrm_claim_count_ge65',
                     'antibiotic_claim_count',
                     'opioid_claim_count',
                     'opioid_day_supply',
                     'opioid_bene_count',
                     'opioid_prescriber_rate' ]

    # assign shorter/simpler names to reamaining columns
    colnames = [ 'npi', 'gender', 'zipcode',
                 'state', 'country', 'specialty', 'medicare_status',
                 'total_day_supply', 'bene_count', 'bene_avg_risk',
                 'antipsych_claims',
                 'hrm_claims',
                 'antibiotic_claims',
                 'op_claims',
                 'op_day_supply',
                 'op_bene_count',
                 'op_rate' ]

    npi = npi.loc[ :, cols_to_keep ]
    npi.columns = colnames

    # ===== data cleaning ======
    if dropna:
        npi = npi.dropna(subset=[ 'op_claims', 'op_day_supply', 'op_bene_count', 'op_rate' ])

    # change the data type of zipcode from float to category
    npi[ 'zipcode' ] = npi[ 'zipcode' ].fillna(0).astype('int').astype('category')

    # omit the non-US samples and drop country feature
    npi = npi.loc[ npi[ 'country' ] == 'US', : ]
    npi = npi.drop('country', 1)

    # drop misc States samples
    misc_states = [ 'XX',  # unknown
                    'ZZ',  # foreign country
                    'AE',  # armed force europe
                    'AP',  # armed force pacific
                    'AA',  # armed force central/south america
                    'AS',  # american samoa
                    'VI',  # virgin island
                    'PR',  # pueto rico
                    'GU',  # guam
                    'MP' ]  # northern mariana islands
    npi = npi[ ~npi[ 'state' ].isin(misc_states) ]

    # drop misc specialty - omit specialties that less than 0.01% of prescribers belong to
    tmp = npi.specialty.value_counts()
    misc_spec = tmp[ tmp < 100 ].index
    npi = npi[ ~npi[ 'specialty' ].isin(misc_spec) ]

    if add_new_features:
        # ===== add new features ======
        npi[ 'op_prescriber' ] = npi[ 'op_bene_count' ] > 0
        npi[ 'avg_op_day_supply' ] = npi[ 'op_day_supply' ] / npi[ 'op_bene_count' ]
        npi[ 'avg_op_day_supply' ] = npi[ 'avg_op_day_supply' ].fillna(0)
        npi[ 'op_longer' ] = npi[ 'avg_op_day_supply' ] > 84  # 12 weeks

    filename2 = filename.split(".")[ 0 ] + "_clean.csv"
    npi.to_csv(filename2)
    logging.info("prescriber summary dataset is prepared and")
    logging.info("saved at {}".format(filename2))

    return npi


def get_drug_name_dict(threshold=500):
    """

    :param threshold: drug names are dropped if number of prescribers is smaller than threshold
    :return: dictionary of generic drug names with five keys 
            - opioid, hrm, antibiotic, antipsych, others
    """
    utils.download_and_decompress(ntl_url, dest_dir)
    filename = dest_dir + "/" + ntl_url.split("/")[ -1 ].split(".")[ 0 ] + ".xlsx"
    ntl = pd.read_excel(filename, sheetname=2, header=1)
    ntl = ntl.loc[ ntl[ 'Number of Prescribers' ] != '  ' ]
    ntl[ 'Number of Prescribers' ] = pd.to_numeric(ntl[ 'Number of Prescribers' ])

    drug_name_dict = {}
    drug_categories = [ 'Opioid Drug Flag',
                        'Antibiotic Drug Flag',
                        'High Risk Medication (HRM) Drug Flag',
                        'Antipsychotic Drug Flag',
                        'others' ]
    for key in drug_categories:
        if key == 'others':
            generic_names = ntl.loc[ (ntl[ 'Opioid Drug Flag' ] == 'N ') \
                                     & (ntl[ 'Antibiotic Drug Flag' ] == 'N ') \
                                     & (ntl[ 'High Risk Medication (HRM) Drug Flag' ] == 'N ') \
                                     & (ntl[ 'Antipsychotic Drug Flag' ] == 'N ') \
                                     & (ntl[ 'Number of Prescribers' ] > threshold),
                                     'Generic Name' ]
        else:
            generic_names = ntl.loc[ (ntl[ key ] == 'Y ') \
                                     & (ntl[ 'Number of Prescribers' ] > threshold),
                                     'Generic Name' ]
        generic_names = utils.clean_txt(generic_names).unique()
        drug_name_dict[ key ] = list(generic_names)

    return ntl, drug_name_dict


def get_drug_names(ntl, drug_name_dict, n_other_drugs=n_other_drugs, random_state=random_state):
    """
    flatten drug_name_dict to get non_opioid names and opioid names lists 
    :param ntl: drug ntl summary table which contains drug names columns 
    :param drug_name_dict: return from get_drug_names_dict()
    :param n_other_drugs: how many ~[op, hrm, antibiotic, antipsych] drugs will be included 
    :return: two lists - op_names and non_op_names
    """

    ntl_small = ntl.sort_values(by='Number of Prescribers', ascending=False)
    ntl_small = ntl_small.loc[ :, [ 'Generic Name', 'Number of Prescribers' ] ]
    ntl_small[ 'Generic Name' ] = utils.clean_txt(ntl_small[ 'Generic Name' ])

    other_names = drug_name_dict[ 'others' ]
    ntl_others = ntl_small.loc[ ntl_small[ 'Generic Name' ].isin(other_names), : ]
    ntl_others[ 'Number of Prescribers' ] = ntl_others[ 'Number of Prescribers' ] / ntl_others[
        'Number of Prescribers' ].sum()

    # randomly sampling subset of other drug names according to its frequency
    np.random.RandomState(seed=random_state)
    others_picked = np.random.choice(ntl_others[ 'Generic Name' ],
                                     size=n_other_drugs,
                                     replace=False,
                                     p=ntl_others[ 'Number of Prescribers' ]).tolist()

    # antibiotic, hrm, and antipsych drugs are all included
    cats_to_flatten = [ 'Antibiotic Drug Flag', 'High Risk Medication (HRM) Drug Flag',
                        'Antipsychotic Drug Flag' ]
    imp_names = [ item for key in cats_to_flatten for item in drug_name_dict[ key ] ]
    non_op_names = others_picked + imp_names
    op_names = drug_name_dict[ 'Opioid Drug Flag' ]
    return non_op_names, op_names


def download_drugs():
    """
    dataset 'detailed data' is downloaded and loaded  
    :return: drugs - TextFileReader object for getting chunks 
    """
    utils.download_and_decompress(drugs_url, dest_dir)
    filename = dest_dir + "/" + drugs_url.split("/")[ -1 ].split(".")[ 0 ] + ".txt"
    drugs = pd.read_table(filename, iterator=True)
    return drugs


def minmaxscaler(X):
    """
    scale a feature so that it has min value 0, and max value 1
    :param X: a pandas series of feature
    :return: scaled feature
    """
    X_std = (X - X.min()) / (X.max() - X.min())
    return X_std


def save_objects(features, labels, i):
    filename = dest_dir + '/batches/batch_' + str(i) + '.pickle'
    with open(filename, 'wb') as f:
        pickle.dump(features, f)
        pickle.dump(labels, f)
        f.close()


def clean_drug_chunks(drugs, npi, non_op_names, op_names, pred_longer=True, chunk_size=chunk_size):
    small = drugs.get_chunk(chunk_size)
    small[ 'avg_day_supply' ] = small[ 'total_day_supply' ] / small[ 'bene_count' ]
    small[ 'generic_name' ] = utils.clean_txt(small[ 'generic_name' ])
    cols_to_keep = [ 'npi', 'generic_name', 'avg_day_supply' ]
    small = small.loc[ :, cols_to_keep ]
    small = small.set_index('npi')

    wide_table = pd.crosstab(index=small.index,
                             columns=small[ 'generic_name' ],
                             values=small[ 'avg_day_supply' ],
                             aggfunc=np.mean)
    drug_names = op_names + non_op_names
    wide_table = wide_table.loc[ :, drug_names ]
    wide_table = wide_table.fillna(0)
    wide_table.index.name = 'npi'

    # =========== feature scaling ==========
    wide_table_sc = wide_table.apply(func=minmaxscaler, axis=1)

    # joined by prescriber summary data
    npi_small = npi.copy()
    cols_to_keep = [ 'npi', 'gender', 'state', 'specialty',
                     'medicare_status', 'op_prescriber', 'op_longer' ]
    npi_small = npi_small.loc[ :, cols_to_keep ]
    npi_small = npi_small.set_index('npi')

    wide_table = wide_table_sc.join(npi_small, how='inner')


    # ========== separate labels from features ==========
    if pred_longer:
        labels = wide_table[ 'op_longer' ]
        features = wide_table.drop([ 'op_longer', 'op_prescriber' ], axis=1)
        features = pd.get_dummies(features)
        # TODO: depending on what samples are in the chunk, the length of one-hot coded
        # will vary - should I fix it?
        return features, labels

    else:
        labels = wide_table[ 'op_prescriber' ]
        cols_to_drop = op_names + [ 'op_longer', 'op_prescriber' ]
        op_features = wide_table.loc[ :, op_names ]
        features = wide_table.drop(cols_to_drop, axis=1)
        features = pd.get_dummies(features)
        return features, op_features, labels


def get_minibatch(drugs, npi, non_op_names, op_names, pred_longer=True, chunk_size=chunk_size):
    features, labels = [ ], [ ]
    try:
        for i in range(n_batches_to_try):
            X, y = clean_drug_chunks(drugs, npi, non_op_names, op_names, pred_longer=True, chunk_size=chunk_size)
            features.append(X)
            labels.append(y)
            save_objects(X, y, i)
            #print(X.shape)
    except StopIteration:
        return None, None
    return features, labels


def main():
    npi = prepare_npi(npi_url, dropna=True, add_new_features=True)
    ntl, drug_name_dict = get_drug_name_dict()
    non_op_names, op_names = get_drug_names(ntl, drug_name_dict)
    drugs = download_drugs()
    features, labels = get_minibatch(drugs, npi, non_op_names, op_names,
                                     pred_longer=True, chunk_size=100000)
    return


if __name__ == '__main__':
    main()
