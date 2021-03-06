import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

class FocalLoss(nn.Module):
    def __init__(self, alpha=1.0, gamma=2.0, logits=False, reduce=False):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.logits = logits
        self.reduce = reduce

    def forward(self, inputs, targets):
        if self.logits:
            BCE_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='sum')
        else:
            loss_fn = nn.BCEWithLogitsLoss(reduction='sum')
            BCE_loss = loss_fn(inputs, targets)

        pt = torch.exp(-BCE_loss)
        F_loss = self.alpha * (1-pt)**self.gamma * BCE_loss

        if self.reduce:
            return torch.mean(F_loss)
        else:
            return F_loss

class FocalCosineLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2, xent=.1, reduction="mean", device="cuda:0"):
        super(FocalCosineLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.xent = xent
        self.reduction = reduction
        self.y = torch.Tensor([1]).to(device)

    def forward(self, input, target):
        target = target.long()
        cosine_loss = F.cosine_embedding_loss(input, target, self.y, reduction=self.reduction)

        cent_loss = F.cross_entropy(F.normalize(input), torch.max(target, 1)[1], reduce=False)
        # cent_loss = self.ce(F.normalize(input), target)
        pt = torch.exp(-cent_loss)
        focal_loss = self.alpha * (1-pt)**self.gamma * cent_loss

        if self.reduction == "mean":
            focal_loss = torch.mean(focal_loss)

        return cosine_loss + self.xent * focal_loss

# Courtesy: https://www.kaggle.com/c/siim-isic-melanoma-classification/discussion/155201
def criterion_margin_focal_binary_cross_entropy(logit, truth):
    weight_pos=2
    weight_neg=1
    gamma=2
    margin=0.2
    em = np.exp(margin)

    logit = logit.view(-1)
    truth = truth.view(-1)
    log_pos = -F.logsigmoid( logit)
    log_neg = -F.logsigmoid(-logit)

    # log_pos = -F.log_softmax( logit)
    # log_neg = -F.log_softmax(-logit)

    log_prob = truth*log_pos + (1-truth)*log_neg
    prob = torch.exp(-log_prob)
    margin = torch.log(em +(1-em)*prob)

    weight = truth*weight_pos + (1-truth)*weight_neg
    loss = margin + weight*(1 - prob) ** gamma * log_prob
    # loss = loss.mean()
    return loss.sum()

# https://gist.github.com/samson-wang/e5cee676f2ae97795356d9c340d1ec7f
def softmax_focal_loss(x, target, gamma=2., alpha=0.25):
    n = x.shape[0]
    device = target.device
    range_n = torch.arange(0, n, dtype=torch.int64, device=device)

    pos_num =  float(x.shape[1])
    p = torch.softmax(x, dim=1)
    p = p[range_n, target]
    loss = -(1-p)**gamma*alpha*torch.log(p)
    return torch.sum(loss) / pos_num

class FocalLossSoftmax(nn.Module):

    def __init__(self, gamma=2, eps=1e-7):
        super(FocalLossSoftmax, self).__init__()
        self.gamma = gamma
        self.eps = eps

    def forward(self, input, target):
        # y = one_hot(target, input.size(-1))
        logit = F.softmax(input, dim=-1)
        logit = logit.clamp(self.eps, 1. - self.eps)

        loss = -1 * target * torch.log(logit) # cross entropy
        loss = loss * (1 - logit) ** self.gamma # focal loss

        return loss.sum()