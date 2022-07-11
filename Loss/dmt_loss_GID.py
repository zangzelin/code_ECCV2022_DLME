# a pytorch based lisv2 code

import pdb
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
        device: Any,
    ):
        super(MyLoss, self).__init__(
            v_input,
            SimilarityFunc=SimilarityFunc,
            device=device)

        self.ITEM_loss = self._GID
        print('use ItakuraSaito loss')
    
    def _GID(self, P, Q):


        EPS = 1e-12
        P_ = P[torch.eye(P.shape[0])==0]*(1-2*EPS) + EPS
        Q_ = Q[torch.eye(P.shape[0])==0]*(1-2*EPS) + EPS
        losssum1 = -1 * (P_ * torch.log(Q_ + EPS)).mean()
        losssum2 = (P_-Q_).mean()
        losssum = losssum1+losssum2

        return losssum/1000
