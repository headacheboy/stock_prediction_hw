from torch.utils.data import RandomSampler, DataLoader, TensorDataset, SequentialSampler
import torch
import numpy as np
from sklearn import preprocessing

name_dict = {"600837.SH": "海通证券", '601318.SH': "中国平安", '600016.SH':"民生银行", '601601.SH':"中国太保",
             '600031.SH': "三一重工", '601688.SH':"华泰证券", '601857.SH':"中国石油", '600276.SH':"恒瑞医药",
             '601988.SH':"中国银行", '600519.SH':"贵州茅台", '601166.SH':"兴业银行", '600028.SH':"中国石化",
             '601628.SH':"中国人寿", '601186.SH':"中国铁建", '601288.SH':"农业银行", '601328.SH':"交通银行",
             '600009.SH':"上海机场", '601398.SH':"工商银行", '601668.SH':"中国建筑", '600048.SH':"保利地产",
             '601939.SH':"建设银行", '600585.SH':"海螺水泥"}

graph = [[0, 1, 3, 5, 12], [2, 8, 10, 14, 15, 17, 20], [4, ], [6, 11], [7], [9], [13, 18, 21], [16], [19]]



batch_size = 16


def load_matrix():
    global graph
    matrix = np.zeros((22, 22))
    for ls in graph:
        for i in range(len(ls)-1):
            for j in range(i+1, len(ls)):
                matrix[i][j] = 1
                matrix[j][i] = 1
    return matrix

def load_data():
    f = open("../data/data", encoding='utf-8').readlines()
    std_date = set()
    for line in f:
        ls = eval(line)
        for ele in ls[1]:
            std_date.add(ele)
    std_date = list(std_date)
    std_date.sort(reverse=False)
    date_length = len(std_date)
    all_price = []
    all_diff = []
    for line in f:
        ls = eval(line)
        std_idx = tmp_idx = 0
        tmp_price = [0 for i in range(date_length)]
        diff = [0 for i in range(date_length)]
        while std_idx < date_length:
            if std_date[std_idx] == ls[1][tmp_idx]:
                tmp_price[std_idx] = ls[2][tmp_idx]
                diff[std_idx] = ls[2][tmp_idx] - ls[3][tmp_idx]
                std_idx += 1
                tmp_idx += 1
            else:
                tmp_price[std_idx] = tmp_price[std_idx-1]
                diff[std_idx] = 0.
                std_idx += 1
        all_price.append(tmp_price)
        all_diff.append(diff)
    return all_price, all_diff

def make_dataset(all_data, diff=False):
    time_window = 11
    data_size = len(all_data[0]) - time_window + 1
    valid_size = int(data_size * 0.8)
    feature_size = len(all_data)
    print(data_size, valid_size)
    train_input_data = np.zeros([valid_size, time_window-1, feature_size])
    valid_input_data = np.zeros([data_size-valid_size, time_window-1, feature_size])
    train_tgt = np.zeros([valid_size, 22])
    valid_tgt = np.zeros([data_size-valid_size, 22])

    scaler = None
    if not diff:
        scaler = preprocessing.StandardScaler()
        all_data = np.array(all_data)
        all_data = scaler.fit_transform(all_data)
        all_data = all_data.tolist()
    for feature_idx in range(len(all_data)):
        feature_ele = all_data[feature_idx]
        for seq_idx in range(len(feature_ele) - 10):
            if seq_idx < valid_size:
                train_tgt[seq_idx][feature_idx] = feature_ele[seq_idx+10]
                for k in range(10):
                    train_input_data[seq_idx][k][feature_idx] = feature_ele[seq_idx + k]
            else:
                valid_tgt[seq_idx-valid_size][feature_idx] = feature_ele[seq_idx+10]
                for k in range(10):
                    valid_input_data[seq_idx-valid_size][k][feature_idx] = feature_ele[seq_idx + k]
    train_dataset = TensorDataset(torch.Tensor(train_input_data), torch.Tensor(train_tgt))
    valid_dataset = TensorDataset(torch.Tensor(valid_input_data), torch.Tensor(valid_tgt))
    train_dataloader = create_dataloader(train_dataset, batch_size, True)
    valid_dataloader = create_dataloader(valid_dataset, batch_size, False)
    return train_dataloader, valid_dataloader, scaler

def create_dataloader(data, batch_size, train):
    if train:
        sampler = RandomSampler(data)
    else:
        sampler = SequentialSampler(data)
    dataloader = DataLoader(data, sampler=sampler, batch_size=batch_size, drop_last=train)
    return dataloader

def get_dataloader(mode="price"):
    price, diff = load_data()
    if mode=="price":
        train_dataloader, valid_dataloader, scaler = make_dataset(price)
    else:
        train_dataloader, valid_dataloader, scaler = make_dataset(diff, True)
    return train_dataloader, valid_dataloader, scaler

if __name__ == '__main__':
    get_dataloader("price")