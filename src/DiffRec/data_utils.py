import numpy as np
from fileinput import filename
import random
import torch
import torch.utils.data as data
import scipy.sparse as sp
import copy
import os
from torch.utils.data import Dataset
import pandas as pd

def data_load(args):
    dataset_path = os.path.abspath(args.data_path+args.dataset)
    file_name = f'{args.dataset}.inter'
    inter_file = os.path.join(dataset_path, file_name)
    uid_field = 'userID'
    iid_field = 'itemID'
    splitting_label = 'x_label'
    field_separator = '\t'
    cols = [uid_field, iid_field, splitting_label]
    df = pd.read_csv(inter_file, usecols=cols, sep=field_separator)

    n_user = df[uid_field].nunique()
    n_item = df[iid_field].nunique()

    dfs = []
    # splitting into training/validation/test
    for i in range(3):
        temp_df = df[df[splitting_label] == i].copy()
        temp_df.drop(splitting_label, inplace=True, axis=1)        # no use again
        dfs.append(temp_df)
    
    train_df, valid_df, test_df = dfs

    train_data = sp.csr_matrix((np.ones_like(train_df[uid_field]), \
            (train_df[uid_field], train_df[iid_field])), dtype='float64', \
            shape=(n_user, n_item))
    valid_y_data = sp.csr_matrix((np.ones_like(valid_df[uid_field]), \
                (valid_df[uid_field], valid_df[iid_field])), dtype='float64',\
                shape=(n_user, n_item))  # valid_groundtruth
    test_y_data = sp.csr_matrix((np.ones_like(test_df[uid_field]), \
                (test_df[uid_field], test_df[iid_field])), dtype='float64', \
                shape=(n_user, n_item))  # test_groundtruth
    

    # train_data = sp.csr_matrix((np.ones_like(train_list[:, 0]), \
    #     (train_list[:, 0], train_list[:, 1])), dtype='float64', \
    #     shape=(n_user, n_item))
    
    # valid_y_data = sp.csr_matrix((np.ones_like(valid_list[:, 0]),
    #              (valid_list[:, 0], valid_list[:, 1])), dtype='float64',
    #              shape=(n_user, n_item))  # valid_groundtruth

    # test_y_data = sp.csr_matrix((np.ones_like(test_list[:, 0]),
    #              (test_list[:, 0], test_list[:, 1])), dtype='float64',
    #              shape=(n_user, n_item))  # test_groundtruth
    
    return train_data, valid_y_data, test_y_data, n_user, n_item


class DataDiffusion(Dataset):
    def __init__(self, data):
        self.data = data
    def __getitem__(self, index):
        item = self.data[index]
        return item
    def __len__(self):
        return len(self.data)
