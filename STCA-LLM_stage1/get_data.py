# -*- coding:utf-8 -*-

import os
import pickle
import random

import numpy as np
import pandas as pd
import torch
import torch_geometric
from pandas import Series
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import Dataset, DataLoader
from torch_geometric import loader
from torch_geometric.data import Data
# from torch_geometric.loader import DataLoader
from torch_geometric.utils import to_undirected
import scipy.sparse as sp
import torch.nn.functional as F
from torch_geometric_temporal import DynamicGraphTemporalSignalBatch
from tqdm import tqdm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def setup_seed(seed):
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


def calc_corr(a, b):
    s1 = Series(a)
    s2 = Series(b)
    return s1.corr(s2)


class MyDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __getitem__(self, item):
        return self.data[item]

    def __len__(self):
        return len(self.data)



def adj2coo(adj):
    # adj numpy
    edge_index_temp = sp.coo_matrix(adj)
    values = edge_index_temp.data
    indices = np.vstack((edge_index_temp.row, edge_index_temp.col))
    edge_index = torch.LongTensor(indices)

    return edge_index


def nn_seq(num_nodes, seq_len, B, pred_step_size):
    data = pd.read_csv(r'D:\pythonproject\STCA-LLM\STCA-LLM_stage1\dataset\wind speed of 20 turbines.csv', header=None)
    # split
    train = data[:int(len(data) * 0.6)]
    val = data[int(len(data) * 0.6):int(len(data) * 0.8)]
    test = data[int(len(data) * 0.8):len(data)]
    # normalization
    scaler = MinMaxScaler()
    train = scaler.fit_transform(data[:int(len(data) * 0.8)].values)
    val = scaler.transform(val.values)
    test = scaler.transform(test.values)


    def process(dataset, batch_size, step_size, shuffle):
        dataset = dataset.tolist()#变成了7027个列表，每个就是一个时刻
        seq = []
        edge_indices = []
        features = []
        targets = []
        edge_weights = []
        batches = []
        ind = 0
        for i in tqdm(range(0, len(dataset) - seq_len - pred_step_size, step_size)):#step_size是每次跳过的步长
            train_seq = []
            for j in range(i, i + seq_len):
                x = []
                for c in range(len(dataset[0])):  # 前24个时刻的所有变量
                    x.append(dataset[j][c])
                train_seq.append(x)
            # 下几个时刻的所有变量
            train_labels = []
            for j in range(len(dataset[0])):
                train_label = []
                for k in range(i + seq_len, i + seq_len + pred_step_size):
                    train_label.append(dataset[k][j])
                train_labels.append(train_label)
            # tensor
            train_seq = torch.FloatTensor(train_seq)
            # print(train_seq.shape)   # 24 13
            train_labels = torch.FloatTensor(train_labels)
            # edge_index = create_graph(num_nodes, train_seq.numpy())
            # graph = Data(x=train_seq.T, edge_index=edge_index, y=train_labels)
            seq.append((train_seq, train_labels))
            # seq.append(graph)


        seq = MyDataset(seq)
        seq = DataLoader(dataset=seq, batch_size=batch_size, shuffle=shuffle, num_workers=0, drop_last=False)

        return seq

    Dtr = process(train, B, step_size=1, shuffle=True)
    Val = process(val, B, step_size=1, shuffle=True)
    Dte = process(test, B, step_size=pred_step_size, shuffle=False)

    return Dtr, Val, Dte, scaler


def save_pickle(dataset, file_name):
    f = open(file_name, "wb")
    pickle.dump(dataset, f)
    f.close()


def load_pickle(file_name):
    f = open(file_name, "rb+")
    dataset = pickle.load(f)
    f.close()
    return dataset
