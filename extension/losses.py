import torch
import torch.nn as nn
from torch import sigmoid, logit

class TrainLoss(nn.Module):
    def __init__(self, alpha=2, beta=4, sigma2=2):
        super(TrainLoss, self).__init__()
        self.alpha = int(alpha)
        self.beta = int(beta)
        self.sigma2 = int(sigma2)

    def forward(self, h0, h1, Phi0, Phi1):
        ''' Modified focal loss. Exactly the same as CornerNet.
            Runs faster and costs a little bit more memory
            Arguments:
            pred (batch x c x h x w)
            gt_regr (batch x c x h x w)
        '''

        logit_Phi0 = logit(Phi0)
        logit_Phi1 = logit(Phi1)

        pos_inds_0 = Phi0.eq(1).float()
        neg_inds_0 = Phi0.lt(1).float()

        pos_inds_1 = Phi0.eq(1).float()
        neg_inds_1 = Phi0.lt(1).float()

        neg_weights_0 = torch.pow(1 - Phi0, self.beta)
        neg_weights_1 = torch.pow(1 - Phi1, self.beta)

        loss = 0.0

        pos_loss_0_focal_term = torch.pow(1 - _sigmoid(h0), self.alpha) * pos_inds_0
        neg_loss_0_focal_term  = torch.pow(_sigmoid(h0), self.alpha) * neg_weights_0 * neg_inds_0

        pos_loss_1_focal_term  = torch.pow(1 - _sigmoid(h1), self.alpha) * pos_inds_1
        neg_loss_1_focal_term  = torch.pow(_sigmoid(h1), self.alpha) * neg_weights_1 * neg_inds_1

        term0 = logit_Phi0 - h0
        term1 = logit_Phi1 - h1

        term0 = term0 * term0
        term1 = term1 * term1

        pos_loss_0 = (pos_loss_0_focal_term * term0).sum()
        neg_loss_0 = (neg_loss_0_focal_term * term0).sum()

        pos_loss_1 = (pos_loss_1_focal_term * term1).sum()
        neg_loss_1 = (neg_loss_1_focal_term * term1).sum()
        
        num_pos_0 = pos_inds_0.float().sum()
        num_pos_1 = pos_inds_1.float().sum()

        if num_pos_0 == 0:
            loss = loss + neg_loss_0
        else:
            loss = loss + (pos_loss_0 + neg_loss_0) / num_pos_0


        if num_pos_1 == 0:
            loss = loss + neg_loss_1
        else:
            loss = loss + (pos_loss_1 + neg_loss_1) / num_pos_1
        
        return loss/self.sigma2


class TestLoss(nn.Module):
    def __init__(self, alpha=2, beta=4):
        super(TestLoss, self).__init__()
        self.alpha = int(alpha)
        self.beta = int(beta)

    def forward(self, h, Phi):
        ''' Modified focal loss. Exactly the same as CornerNet.
            Runs faster and costs a little bit more memory
            Arguments:
            pred (batch x c x h x w)
            gt_regr (batch x c x h x w)
        '''

        h = _sigmoid(h)

        pos_inds = Phi.eq(1).float()
        neg_inds = Phi.lt(1).float()

        neg_weights = torch.pow(1 - Phi, self.beta)

        loss = 0.0

        pos_loss = torch.log(h) * torch.pow(1 - h, self.alpha) * pos_inds
        neg_loss = torch.log(1 - h) * torch.pow(h, self.alpha) * neg_weights * neg_inds

        num_pos  = pos_inds.float().sum()
        pos_loss = pos_loss.sum()
        neg_loss = neg_loss.sum()

        if num_pos == 0:
            loss = loss - neg_loss
        else:
            loss = loss - (pos_loss + neg_loss) / num_pos 
            
        return loss



def _sigmoid(x):
    y = torch.clamp(x.sigmoid_(), min=1e-4, max=1-1e-4)
    return y


def _logit(x):
    y = torch.clamp(logit(x), min=logit(torch.tensor(1e-4)), max=logit(torch.tensor(1-1e-4)))
    return y