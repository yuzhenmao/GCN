import torch
import torch.nn as nn
import torch.nn.functional as F
import pdb

class link_pred_loss(nn.Module):
    def __int__(self):
        super(link_pred_loss, self).__int__()

    def forward(self, emb_u, emb_v, emb_neg_v):
        score = torch.sum(torch.mul(emb_u.squeeze(1), emb_v.squeeze(1)), dim=1)
        # score = torch.clamp(score, max=10, min=-10)
        score = -F.logsigmoid(score)

        neg_score = torch.bmm(emb_neg_v, emb_u.view(-1, emb_neg_v.shape[2], 1)).squeeze()
        # neg_score = torch.clamp(neg_score, max=10, min=-10)
        neg_score = -torch.sum(F.logsigmoid(-neg_score), dim=1)

        return torch.mean(score + neg_score)
