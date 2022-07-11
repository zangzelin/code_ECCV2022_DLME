# a pytorch based lisv2 code

from multiprocessing import Pool

import numpy as np
import torch
import torch.autograd
import torch.nn.functional as F
from scipy import optimize
from torch import nn
from torch.autograd import Variable
from torch.functional import split
from torch.nn.modules import loss
from typing import Any
import scipy

from Loss.dmt_loss_source import Source

class MyLoss(Source):
    def __init__(
        self,
        v_input,
        SimilarityFunc,
        metric="braycurtis",
    ):
        super(MyLoss, self).__init__(
            v_input,
            SimilarityFunc=SimilarityFunc,
            metric=metric
            )

        self.ITEM_loss = self._TwowaydivergenceLoss
    
    def _TwowaydivergenceLoss(self, P, Q):

        EPS = 1e-12
        P_ = P[torch.eye(P.shape[0])==0]*(1-2*EPS) + EPS
        Q_ = Q[torch.eye(P.shape[0])==0]*(1-2*EPS) + EPS
        losssum1 = (P_ * torch.log(Q_ + EPS)).mean()
        losssum2 = ((1-P_) * torch.log(1-Q_ + EPS)).mean()
        losssum = -1*(losssum1 + losssum2)

        if torch.isnan(losssum):
            input('stop and find nan')
        return losssum
