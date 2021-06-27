"""
Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
Re-implement by Yuzhen Mao
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.nn.parameter import Parameter


class GraphConv(torch.nn.Module):

    def __init__(self, in_features, out_features):
        super(GraphConv, self).__init__()
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        self.bias = Parameter(torch.FloatTensor(out_features))

    def forward(self, input, adj):
        support = torch.mm(input, self.weight)
        output = torch.spmm(adj, support)
        return output + self.bias


class MY_GCN(torch.nn.Module):

    def __init__(self, nfeat, nhid, nclass, ndim, depth, bias=True, dropout=0.5):
        super(MY_GCN, self).__init__()
        self.depth = depth
        self.bias = bias
        self.dropout = dropout

        gc = []
        for i in range(depth + 2):
            if i == 0:
                gc.append(GraphConv(nfeat, nhid[0]))
            elif i == depth:
                gc.append(GraphConv(nhid[depth-1], nclass))
            elif i == depth + 1:
                gc.append(GraphConv(nhid[depth-1], ndim))
            else:
                gc.append(GraphConv(nhid[i-1], nhid[i]))
        self.gc = nn.ModuleList(gc)
        self.randominit()

    def randominit(self):
        for i in range(self.depth + 2):
            out_dim, in_dim = self.gc[i].weight.shape
            stdv = np.sqrt(2 / (in_dim + out_dim))
            self.gc[i].weight.data.uniform_(-stdv, stdv)
            # self.gc[i].weight.data.normal_(0, stdv)
            if self.bias is not None:
                self.gc[i].bias.data.uniform_(-stdv, stdv)
            else:
                self.gc[i].bias.data.fill_(0.0)
            # stdv = 1. / math.sqrt(self.weight.size(1))
            # self.weight.data.uniform_(-stdv, stdv)

    def forward(self, H, A):
        for i in range(0, self.depth):
            H = self.gc[i](H, A)
            H = F.relu(H)
            H = F.dropout(H, self.dropout, training=self.training)
        H_class = self.gc[self.depth](H, A)
        H_link = self.gc[self.depth + 1](H, A)
        # return F.log_softmax(H, dim=1)
        return H_class, H_link
