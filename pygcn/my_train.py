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

from pygcn.utils.utils import load_data, class_accuracy, my_load_data, link_accuracy, test_class_accuracy
from pygcn.utils.data_reader import MyDataset, MyData
from pygcn.models import GCN
from pygcn.model.gcn import MY_GCN
from pygcn.model.combined_loss import CombinedLoss

# Training settings
parser = argparse.ArgumentParser()
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='Disables CUDA training.')
parser.add_argument('--fastmode', action='store_true', default=False,
                    help='Validate during training pass.')
parser.add_argument('--seed', type=int, default=42, help='Random seed.')
parser.add_argument('--epochs', type=int, default=100,
                    help='Number of epochs to train.')
parser.add_argument('--batch_size', type=int, default=16,
                    help='Batch_size to train.')
parser.add_argument('--lr', type=float, default=0.01,
                    help='Initial learning rate.')
parser.add_argument('--weight_decay', type=float, default=5e-4,
                    help='Weight decay (L2 loss on parameters).')
parser.add_argument('--hidden', type=list, default=[16, 64, 128],
                    help='Number of hidden units.')
parser.add_argument('--dim', type=int, default=50,
                    help='Number of link-prediction dim.')
parser.add_argument('--depth', type=int, default=1,
                    help='Number of layers. (exclude the last layer)')
parser.add_argument('--dropout', type=float, default=0.5,
                    help='Dropout rate (1 - keep probability).')
parser.add_argument('--weight', type=str, default=None,
                    help='Load weights.')

args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

np.random.seed(args.seed)
torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)

# Load data
adj, actual_adj, features, labels, conn, conn_hide, un_conn, frequency = my_load_data(load=True)
data = MyData(adj, actual_adj, features, labels, conn, conn_hide, un_conn, frequency)
dataset = MyDataset(data, "train", neg_num=15)
dataloader = DataLoader(dataset, batch_size=args.batch_size,
                                     shuffle=False, num_workers=0, collate_fn=dataset.collate)
nclass = labels.max().item() + 1

# Model and optimizer
model = MY_GCN(nfeat=features.shape[1],
            nhid=args.hidden,
            nclass=nclass,
            ndim=args.dim,
            depth=args.depth,
            bias=True,
            dropout=args.dropout)
loss_fn = CombinedLoss()
gcn_optimizer = optim.Adam(model.parameters(), lr=args.lr)
gcn_scheduler = torch.optim.lr_scheduler.MultiStepLR(gcn_optimizer, milestones=[40, 60, 80, 90], gamma=0.5, last_epoch=-1)
loss_optimizer = optim.Adam(loss_fn.parameters(),
                       lr=args.lr, weight_decay=args.weight_decay)
# loss_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(loss_optimizer, len(dataloader))
loss_scheduler = torch.optim.lr_scheduler.MultiStepLR(loss_optimizer, milestones=[40, 60, 80, 90], gamma=0.5, last_epoch=-1)

if args.cuda:
    model.cuda()
    loss_fn.cuda()
    features = features.cuda()
    adj = adj.cuda()
    labels = labels.cuda()

epoch = 0
# args.weight = "./checkpint_99_epoch_bi.pkl"
if args.weight is not None:
    print('Recovering from %s ...' % (args.weight))
    checkpoint = torch.load(args.weight)
    epoch = checkpoint["epoch"]

    model.load_state_dict(checkpoint["model_state_dict"])
    loss_fn.load_state_dict(checkpoint["loss_state_dict"])
    gcn_optimizer.load_state_dict(checkpoint["optimizer_state_dic"])
    # gcn_scheduler.load_state_dict(checkpoint["schedule_state_dic"])
    loss_optimizer.load_state_dict(checkpoint["loss_optimizer_state_dic"])
    # loss_scheduler.load_state_dict(checkpoint["loss_schedule_state_dic"])

n_batches = len(dataloader)


def train(epoch):
    for batch_idx, sample_batched in enumerate(dataloader):

        pos_u_idx = sample_batched[0].cuda()
        pos_v_idx = sample_batched[1].cuda()
        neg_v_idx = sample_batched[2].cuda()

        idx_train = torch.cat([pos_u_idx, pos_v_idx, neg_v_idx], 1)  # used for classification task
        new_idx = torch.where(idx_train < 1500)
        # print(len(new_idx[0]))

        t = time.time()
        model.train()
        gcn_optimizer.zero_grad()
        loss_optimizer.zero_grad()
        output_class, output_link = model(features, adj)
        class_loss, link_loss, loss_train = loss_fn(output_class[idx_train][new_idx].view(-1, nclass),
                labels[idx_train][new_idx].view(-1), output_class[pos_u_idx], output_class[pos_v_idx], output_class[neg_v_idx])
        loss_train.backward()
        print("lr: ", gcn_scheduler.get_last_lr())
        print("loss train:", loss_train)
        # print("alpha: ", loss_fn.alpha)
        # print("beta: ", loss_fn.beta)
        class_acc_train = class_accuracy(output_class[idx_train], labels[idx_train])
        link_acc_train = link_accuracy(output_class[pos_u_idx], output_class[pos_v_idx], output_class[neg_v_idx])
        # for name, parms in model.named_parameters():
        #     print('-->name:', name, '-->grad_requirs:', parms.requires_grad, ' -->grad_value:', parms.grad)
        gcn_optimizer.step()
        loss_optimizer.step()

        if not args.fastmode:
            # Evaluate validation set performance separately,
            # deactivates dropout during validation run.
            model.eval()
            output = model(features, adj)

        # loss_val = F.nll_loss(output[idx_val], labels[idx_val])
        # loss_val = class_loss_fn(output[idx_val], labels[idx_val]) + link_loss_fn(output[idx_val])
        # acc_val = class_accuracy(output[idx_val], labels[idx_val])
        print('Epoch: {:04d}'.format(epoch+1),
              'Batch: {:d}/{:d}'.format(batch_idx, n_batches),
              'class_loss_train: {:.4f}'.format(class_loss.item()),
              'link_loss_train: {:.4f}'.format(link_loss.item()),
              'class_acc_train: {:.4f}'.format(class_acc_train.item()),
              'link_acc_train: {:.4f}'.format(link_acc_train),
              # 'loss_val: {:.4f}'.format(loss_val.item()),
              # 'acc_val: {:.4f}'.format(acc_val.item()),
              'time: {:.4f}s'.format(time.time() - t))

        if epoch == args.epochs-1:
            checkpoint = {"model_state_dict": model.state_dict(),
                          "loss_state_dict": loss_fn.state_dict(),
                          "optimizer_state_dic": gcn_optimizer.state_dict(),
                          "schedule_state_dic": gcn_scheduler.state_dict(),
                          "loss_optimizer_state_dic": loss_optimizer.state_dict(),
                          "loss_schedule_state_dic": loss_scheduler.state_dict(),
                          "loss": loss_train,
                          "epoch": epoch}
            path_checkpoint = "./checkpint_{}_epoch_bi.pkl".format(epoch)
            torch.save(checkpoint, path_checkpoint)


def test(weight):
    if weight is not None:
        print('Recovering from %s ...' % (weight))
        checkpoint = torch.load(weight)
        epoch = checkpoint["epoch"]

        model.load_state_dict(checkpoint["model_state_dict"])
        loss_fn.load_state_dict(checkpoint["loss_state_dict"])

    model.eval()
    output_class, output_link = model(features, adj)
    dataset.change_task("test")
    dataloader = DataLoader(dataset, batch_size=128,
                            shuffle=False, num_workers=0, collate_fn=dataset.collate)
    for i, sample_batched in enumerate(dataloader):
        pos_u_idx = sample_batched[0].cuda()
        pos_v_idx = sample_batched[1].cuda()
        neg_v_idx = sample_batched[2].cuda()
        idx_test = torch.LongTensor(range(1500, adj.shape[0])).cuda()
        class_loss, link_loss, loss_train = loss_fn(output_class[idx_test].view(-1, nclass),
                                                    labels[idx_test].view(-1), output_class[pos_u_idx],
                                                    output_class[pos_v_idx], output_class[neg_v_idx])
        class_acc_test = test_class_accuracy(output_class[idx_test], labels[idx_test])
        link_acc_test = link_accuracy(output_class[pos_u_idx], output_class[pos_v_idx], output_class[neg_v_idx])
        print('class_loss_train: {:.4f}'.format(class_loss.item()),
              'link_loss_train: {:.4f}'.format(link_loss.item()),
              'class_acc_train: {:.4f}'.format(class_acc_test.item()),
              'link_acc_train: {:.4f}'.format(link_acc_test))


# Train model
t_total = time.time()
for epoch in range(epoch, args.epochs):
    train(epoch)
    gcn_scheduler.step()
    loss_scheduler.step()
print("Optimization Finished!")
print("Total time elapsed: {:.4f}s".format(time.time() - t_total))

# Testing
test("./checkpint_99_epoch_bi.pkl")



