import numpy as np
import torch
from torch.utils.data import Dataset

np.random.seed(12345)


class MyData:
    NEGATIVE_TABLE_SIZE = 1e8

    def __init__(self, adj, actual_adj, features, labels, conn, hide_conn, un_conn, frequency):

        self.adj = adj
        self.actual_adj = actual_adj
        self.features = features
        self.labels = labels
        self.conn = conn
        self.hide_conn = hide_conn
        self.un_conn = un_conn[['node_1', "node_2"]].values.tolist()
        self.negatives = []
        self.discards = []
        self.negpos = 0

        self.link_count = len(self.conn)
        self.token_count = adj.shape[0]
        self.word_frequency = frequency

        self.initTableNegatives()
        # self.initTableDiscards()

    # def initTableDiscards(self):
    #     t = 0.0001
    #     f = np.array(list(self.word_frequency)) / self.token_count
    #     self.discards = np.sqrt(t / f) + (t / f)

    def initTableNegatives(self):
        pow_frequency = np.array(list(self.word_frequency)) ** 0.5
        words_pow = sum(pow_frequency)
        ratio = pow_frequency / words_pow
        count = np.round(ratio * MyData.NEGATIVE_TABLE_SIZE)
        for wid, c in enumerate(count):
            self.negatives += [wid] * int(c)
        self.negatives = np.array(self.negatives)
        np.random.shuffle(self.negatives)

    def getNegatives(self, target, size):
        response = []
        while len(response) < size:
            j = self.negatives[self.negpos]
            if self.actual_adj[target, j] == 0:
                response.append(j)
            if self.negpos == (len(self.negatives) - 1):
                self.negpos = 0
            else:
                self.negpos = self.negpos + 1
        return response


# -----------------------------------------------------------------------------------------------------------------
class MyDataset(Dataset):
    def __init__(self, data, task, neg_num):
        self.data = data
        self.task = task
        self.neg_num = neg_num

    def __len__(self):
        if self.task == "train":
            return self.data.link_count
        if self.task == "test":
            return len(self.data.hide_conn)

    def __getitem__(self, idx):
        if self.task == "train":
            links = self.data.conn.loc[idx]
            return [links['node_1'], links['node_2'], self.data.getNegatives(links['node_1'], self.neg_num)]
        if self.task == "test":
            links = self.data.hide_conn.loc[idx]
            return [links['node_1'], links['node_2'], self.data.getNegatives(links['node_1'], self.neg_num)]

    def change_task(self, task):
        self.task = task

    def collate(self, batches):
        all_u_idx = [batch[0] for batch in batches if len(batch) > 0]
        all_v_idx = [batch[1] for batch in batches if len(batch) > 0]
        all_neg_v_idx = [batch[2] for batch in batches if len(batch) > 0]
        all_u_idx = np.array(all_u_idx).reshape(-1, 1)
        all_v_idx = np.array(all_v_idx).reshape(-1, 1)
        all_neg_v_idx = np.array(all_neg_v_idx).reshape(-1, self.neg_num)

        return torch.LongTensor(all_u_idx), torch.LongTensor(all_v_idx), torch.LongTensor(all_neg_v_idx)

