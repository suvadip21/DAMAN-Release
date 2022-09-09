
import torch
import torch.nn as nn
import torch.nn.functional as F
from distutils.version import LooseVersion


class DiceLoss(nn.Module):
    def forward(self, pred, target, mu=1.0):
        pred = pred.contiguous()
        target = target.contiguous()
        intersection = (pred * target).sum(dim=2).sum(dim=2)
        loss = (1 - ((2. * intersection + mu) / (pred.sum(dim=2).sum(dim=2) + target.sum(dim=2).sum(dim=2) + mu)))
        return loss.mean() * 10

class BCELoss(nn.Module):
    def forward(self, x, y):
       """
       x: input of size [n_b, ...]
       y: label of same size as x, should be in  [0, 1]
       L = [l1, ..., ln, ... lN], where ln = -ynlog(xn) - (1-yn)log(1-xn) 
       """ 
       y = y.contiguous()
       x = x.contiguous()
    #    import pdb; pdb.set_trace() 
       loss = torch.nn.BCEWithLogitsLoss()(x, y)
       return loss 


class KLDLoss(nn.Module):

    def forward(self, mu, logvar):

        batch_size = mu.size(0)
        assert batch_size != 0
        if mu.data.ndimension() == 4:
            mu = mu.view(mu.size(0), mu.size(1))
        if logvar.data.ndimension() == 4:
            logvar = logvar.view(logvar.size(0), logvar.size(1))

        klds = -0.5*(1 + logvar - mu.pow(2) - logvar.exp())
        total_kld = klds.sum(1).mean(0, True)
        dimension_wise_kld = klds.mean(0)
        mean_kld = klds.mean(1).mean(0, True)

        return total_kld, dimension_wise_kld, mean_kld

class CrossEntropyLoss(nn.Module):

    def __init__(self, weight=None, size_average=False):
        super(CrossEntropyLoss, self).__init__()

        self.weight = weight
        self.size_average = size_average

    def forward(self, input, target):

        # input: (n, c, h, w), target: (n, h, w)
        input_size = input.size()
        if len(input_size) == 2: # 1D
            n, c = input.size()
        elif len(input_size) == 4: # 2D
            n, c, h, w = input.size()
        else: # 3D
            n, c, h, w, d = input.size()
        # log_p: (n, c, h, w)
        if LooseVersion(torch.__version__) < LooseVersion('0.3'):
            # ==0.2.X
            log_p = F.log_softmax(input)
        else:
            # >=0.3
            log_p = F.log_softmax(input, dim=1)
        # log_p: (n*h*w, c)
        if len(input_size) == 2: # 1D
            log_p = log_p.contiguous()
            log_p = log_p[target.view(n, 1).repeat(1, c) >= 0]
        elif len(input_size) == 4: # 2D
            log_p = log_p.transpose(1, 2).transpose(2, 3).contiguous()
            log_p = log_p[target.view(n, h, w, 1).repeat(1, 1, 1, c) >= 0]
        else: # 3D
            log_p = log_p.transpose(1, 2).transpose(2, 3).transpose(3, 4).contiguous()
            log_p = log_p[target.view(n, h, w, d, 1).repeat(1, 1, 1, 1, c) >= 0]
        log_p = log_p.view(-1, c)
        # target: (n*h*w,)
        mask = target >= 0
        target = target[mask]
        # import pdb; pdb.set_trace()
        loss = F.nll_loss(log_p, target.long(), weight=self.weight)
        if self.size_average:
            loss /= mask.data.sum()
        return loss

class MSELoss(nn.Module):

    def forward(self, input, target):

        return torch.mean((input-target)**2)