import numpy as np
import scipy.sparse as sp
import torch
import pandas as pd
import networkx as nx
from tqdm import tqdm
import torch.nn.functional as F

np.random.seed(12345)


def encode_onehot(labels):
    classes = set(labels)
    classes_dict = {c: np.identity(len(classes))[i, :] for i, c in
                    enumerate(classes)}
    labels_onehot = np.array(list(map(classes_dict.get, labels)),
                             dtype=np.int32)
    return labels_onehot


def load_data(path="../data/cora/", dataset="cora"):
    """Load citation network dataset (cora only for now)"""
    print('Loading {} dataset...'.format(dataset))

    idx_features_labels = np.genfromtxt("{}{}.content".format(path, dataset),
                                        dtype=np.dtype(str))
    features = sp.csr_matrix(idx_features_labels[:, 1:-1], dtype=np.float32)
    labels = encode_onehot(idx_features_labels[:, -1])

    # build graph
    idx = np.array(idx_features_labels[:, 0], dtype=np.int32)
    idx_map = {j: i for i, j in enumerate(idx)}
    edges_unordered = np.genfromtxt("{}{}.cites".format(path, dataset),
                                    dtype=np.int32)
    edges = np.array(list(map(idx_map.get, edges_unordered.flatten())),
                     dtype=np.int32).reshape(edges_unordered.shape)
    adj = sp.coo_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])),
                        shape=(labels.shape[0], labels.shape[0]),
                        dtype=np.float32)

    # build symmetric adjacency matrix
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)

    features = normalize(features)
    adj = normalize(adj + sp.eye(adj.shape[0]))

    idx_train = range(140)
    idx_val = range(200, 500)
    idx_test = range(500, 1500)

    features = torch.FloatTensor(np.array(features.todense()))
    labels = torch.LongTensor(np.where(labels)[1])
    adj = sparse_mx_to_torch_sparse_tensor(adj)

    idx_train = torch.LongTensor(idx_train)
    idx_val = torch.LongTensor(idx_val)
    idx_test = torch.LongTensor(idx_test)

    return adj, features, labels, idx_train, idx_val, idx_test


def my_load_data(path="../data/cora/", dataset="cora", load=False):
    """Load citation network dataset (cora only for now)"""
    print('Loading {} dataset...'.format(dataset))

    if load is True:
        adj = sp.load_npz("adj.npz")
        actual_adj = sp.load_npz("actual_adj.npz")
        features = sp.load_npz("features.npz")
        labels = np.load("labels.npy")
        frequency = np.load("frequency.npy")
        conn = pd.read_csv("conn.csv")
        conn_hide = pd.read_csv("conn_hide.csv")
        un_conn = pd.read_csv("un_conn.csv")
    else:
        idx_features_labels = np.genfromtxt("{}{}.content".format(path, dataset),
                                            dtype=np.dtype(str))
        features = sp.csr_matrix(idx_features_labels[:, 1:-1], dtype=np.float32)
        labels = encode_onehot(idx_features_labels[:, -1])

        # build graph
        idx = np.array(idx_features_labels[:, 0], dtype=np.int32)
        idx_map = {j: i for i, j in enumerate(idx)}
        edges_unordered = np.genfromtxt("{}{}.cites".format(path, dataset),
                                        dtype=np.int32)
        edges = np.array(list(map(idx_map.get, edges_unordered.flatten())),
                         dtype=np.int32).reshape(edges_unordered.shape)
        adj = sp.coo_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])),
                            shape=(labels.shape[0], labels.shape[0]),
                            dtype=np.float32)

        # build symmetric adjacency matrix
        adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
        frequency = np.sum(adj.todense(), axis=1)

        # get unconnected node-pairs
        half_adj = np.tril(np.mat(adj + np.ones(adj.shape)), -1)
        un_conn_x, un_conn_y = np.nonzero(half_adj == 1)
        un_conn = pd.DataFrame({'node_1': un_conn_x,
                             'node_2': un_conn_y})
        un_conn['link'] = 0
        conn_x, conn_y = np.nonzero(half_adj > 1)
        conn = pd.DataFrame({'node_1': conn_x,
                             'node_2': conn_y})
        conn['link'] = 1
        # all_connected_pairs = list(zip(conn_x, conn_y))
        # print(len(all_unconnected_pairs))
        # print(len(all_connected_pairs))

        ###############################################
        # if the original graph is connected
        # store removable links
        conn_temp = conn.copy()
        omissible_links_index = []
        G_temp = nx.from_pandas_edgelist(conn_temp, "node_1", "node_2", create_using=nx.Graph())
        connected = nx.number_connected_components(G_temp)
        for i in tqdm(conn.index.values):
            # remove a node pair and build a new graph
            G_temp.remove_edge(conn_x[i], conn_y[i])
            # check there is no spliting of graph and number of nodes is same
            if (nx.number_connected_components(G_temp) == connected) and (len(G_temp.nodes) == adj.shape[0]):
                omissible_links_index.append(i)
                conn_temp = conn_temp.drop(index=i)
            else:
                G_temp.add_edge(conn_x[i], conn_y[i])
        # print(len(omissible_links_index))   2648 / 5278
        if len(omissible_links_index) * 2 > len(conn_x):
            omissible_links_index = list(np.random.choice(omissible_links_index, len(conn_x) // 2, replace=False))
        conn_hide = conn.loc[omissible_links_index]
        # add the target variable 'link'
        conn_hide['link'] = 1
        actual_adj = adj.copy()
        for i in omissible_links_index:
            adj[conn_x[i], conn_y[i]] = 0
            adj[conn_y[i], conn_x[i]] = 0
            actual_adj[conn_x[i], conn_y[i]] = -1
            actual_adj[conn_y[i], conn_x[i]] = -1
            conn = conn.drop(index=i)
        ###############################################

        # edge_data = un_conn.append(conn_hide[['node_1', 'node_2', 'link']], ignore_index=True)
        # print(edge_data['link'].value_counts())   3,660,000 : 2648 / 5278

        conn_cp = conn.copy()
        conn_cp[['node_1', 'node_2']] = conn[['node_2', 'node_1']]
        conn = conn.append(conn_cp[['node_1', 'node_2', 'link']], ignore_index=True)

        features = normalize(features)
        adj = normalize(adj + sp.eye(adj.shape[0]))
        sp.save_npz("adj.npz", adj)
        sp.save_npz("actual_adj.npz", actual_adj)
        sp.save_npz("features.npz", features)
        np.save("labels.npy", labels)
        np.save("frequency.npy", frequency)
        conn.to_csv("conn.csv")
        conn_hide.to_csv("conn_hide.csv")
        un_conn.to_csv("un_conn.csv")

    features = torch.FloatTensor(np.array(features.todense()))
    labels = torch.LongTensor(np.where(labels)[1])
    adj = sparse_mx_to_torch_sparse_tensor(adj)
    actual_adj = sparse_mx_to_torch_sparse_tensor(actual_adj)


    return adj, actual_adj, features, labels, conn.reset_index(drop=True), conn_hide.reset_index(drop=True), un_conn, frequency

def normalize(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx


def class_accuracy(output, labels):
    preds = output.max(2)[1].type_as(labels)
    correct = preds.eq(labels).double()
    correct = correct.sum()
    return correct / (labels.shape[0]*labels.shape[1])

def test_class_accuracy(output, labels):
    preds = output.max(1)[1].type_as(labels)
    correct = preds.eq(labels).double()
    correct = correct.sum()
    return correct / len(labels)


def link_accuracy(emb_u, emb_v, emb_neg_v):
    score = torch.sum(torch.mul(emb_u.squeeze(1), emb_v.squeeze(1)), dim=1)
    correct = len(torch.where(score > 0)[0])

    neg_score = torch.bmm(emb_neg_v, emb_u.view(-1, emb_neg_v.shape[2], 1)).squeeze()
    correct = correct + len(torch.where(neg_score < 0)[0])

    return correct / (neg_score.shape[0]*neg_score.shape[1] + score.shape[0])


def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)
