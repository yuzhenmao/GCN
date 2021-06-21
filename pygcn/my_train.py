"""
Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
Re-implement by Yuzhen Mao
"""

from __future__ import division
from __future__ import print_function

import time
import argparse
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torch.optim as optim

from pygcn.utils.utils import load_data, accuracy, my_load_data
from pygcn.utils.data_reader import MyDataset, MyData
from pygcn.models import GCN
from pygcn.model.gcn import MY_GCN
from pygcn.model.link_pred_loss import link_pred_loss

# Training settings
parser = argparse.ArgumentParser()
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='Disables CUDA training.')
parser.add_argument('--fastmode', action='store_true', default=False,
                    help='Validate during training pass.')
parser.add_argument('--seed', type=int, default=42, help='Random seed.')
parser.add_argument('--epochs', type=int, default=200,
                    help='Number of epochs to train.')
parser.add_argument('--batch_size', type=int, default=8,
                    help='Batch_size to train.')
parser.add_argument('--lr', type=float, default=0.01,
                    help='Initial learning rate.')
parser.add_argument('--weight_decay', type=float, default=5e-4,
                    help='Weight decay (L2 loss on parameters).')
parser.add_argument('--hidden', type=int, default=16,
                    help='Number of hidden units.')
parser.add_argument('--dropout', type=float, default=0.5,
                    help='Dropout rate (1 - keep probability).')

args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

np.random.seed(args.seed)
torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)

# Load data
adj, features, labels, conn, conn_hide, un_conn, frequency = my_load_data()
data = MyData(adj, features, labels, conn, conn_hide, un_conn, frequency)
dataset = MyDataset(data, "train", 5)
dataloader = DataLoader(dataset, batch_size=args.batch_size,
                                     shuffle=False, num_workers=0, collate_fn=dataset.collate)

# Model and optimizer
model = MY_GCN(nfeat=features.shape[1],
            nhid=args.hidden,
            nclass=labels.max().item() + 1,
            depth=1,
            bias=True,
            dropout=args.dropout)
optimizer = optim.Adam(model.parameters(),
                       lr=args.lr, weight_decay=args.weight_decay)
class_loss_fn = nn.CrossEntropyLoss()
link_loss_fn = link_pred_loss()

if args.cuda:
    model.cuda()
    features = features.cuda()
    adj = adj.cuda()
    labels = labels.cuda()


def train(epoch):
    for i, sample_batched in enumerate(dataloader):

        pos_u_idx = sample_batched[0].cuda()
        pos_v_idx = sample_batched[1].cuda()
        neg_v_idx = sample_batched[2].cuda()

        idx_train = torch.cat([pos_u_idx, pos_v_idx, neg_v_idx], 1)  # used for classification task

        t = time.time()
        model.train()
        optimizer.zero_grad()
        output = model(features, adj)
        class_loss = class_loss_fn(output[idx_train], labels[idx_train])
        link_loss = link_loss_fn(output[pos_u_idx], output[pos_v_idx], output[neg_v_idx])
        loss_train = class_loss + link_loss
        acc_train = accuracy(output[idx_train], labels[idx_train])
        loss_train.backward()
        optimizer.step()

        if not args.fastmode:
            # Evaluate validation set performance separately,
            # deactivates dropout during validation run.
            model.eval()
            output = model(features, adj)

        # loss_val = F.nll_loss(output[idx_val], labels[idx_val])
        # loss_val = class_loss_fn(output[idx_val], labels[idx_val]) + link_loss_fn(output[idx_val])
        # acc_val = accuracy(output[idx_val], labels[idx_val])
        print('Epoch: {:04d}'.format(epoch+1),
              'loss_train: {:.4f}'.format(loss_train.item()),
              'acc_train: {:.4f}'.format(acc_train.item()),
              # 'loss_val: {:.4f}'.format(loss_val.item()),
              # 'acc_val: {:.4f}'.format(acc_val.item()),
              'time: {:.4f}s'.format(time.time() - t))


def test():
    model.eval()
    output = model(features, adj)
    dataset.change_task("test")
    for i, sample_batched in enumerate(dataloader):
        if args.cuda:
            pos_u = sample_batched[0].cuda()
            pos_v = sample_batched[1].cuda()
            neg_v = sample_batched[2].cuda()
        # loss_test = F.nll_loss(output[idx_test], labels[idx_test])
        loss_test = class_loss_fn(output[idx_test], labels[idx_test]) + link_loss_fn(output[idx_test])
        acc_test = accuracy(output[idx_test], labels[idx_test])
        print("Test set results:",
              "loss= {:.4f}".format(loss_test.item()),
              "accuracy= {:.4f}".format(acc_test.item()))


# Train model
t_total = time.time()
for epoch in range(args.epochs):
    train(epoch)
print("Optimization Finished!")
print("Total time elapsed: {:.4f}s".format(time.time() - t_total))

# Testing
# test()



