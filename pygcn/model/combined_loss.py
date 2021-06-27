import torch
import torch.nn as nn
import torch.nn.functional as F
import pdb
from torch.nn.parameter import Parameter


def link_pred_loss(emb_u, emb_v, emb_neg_v):
    score = torch.sum(torch.mul(emb_u.squeeze(1), emb_v.squeeze(1)), dim=1)
    score = torch.clamp(score, max=10, min=-10)
    score = -F.logsigmoid(score)
    # print(score)

    neg_score = torch.bmm(emb_neg_v, emb_u.view(-1, emb_neg_v.shape[2], 1)).squeeze()
    neg_score = torch.clamp(neg_score, max=10, min=-10)
    neg_score = -torch.sum(F.logsigmoid(-neg_score), dim=1)
    # print(neg_score)

    return torch.mean(score + neg_score)


class CombinedLoss(torch.nn.Module):

    def __init__(self):
        super(CombinedLoss, self).__init__()
        self.class_loss_fn = nn.CrossEntropyLoss()
        self.alpha = Parameter(data=torch.FloatTensor(1))
        self.alpha.data.uniform_(1, 1)
        self.beta = Parameter(data=torch.FloatTensor(1))
        self.beta.data.uniform_(1, 1)

    def forward(self, emb_all, labels, emb_u, emb_v, emb_neg_v):
        class_loss = self.class_loss_fn(emb_all, labels)
        link_loss = link_pred_loss(emb_u, emb_v, emb_neg_v)
        combined_loss = (1.0/(2.0*self.alpha*self.alpha)) * class_loss + \
                        (1.0/(2.0*self.beta*self.beta)) * link_loss + torch.log(self.alpha*self.beta)

        return class_loss, link_loss, combined_loss
