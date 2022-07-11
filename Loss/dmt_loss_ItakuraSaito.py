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

        self.ITEM_loss = self._ItakuraSaito
        print('use ItakuraSaito loss')
    
    def _ItakuraSaito(self, P, Q):


        PdQ = (P)/(Q+ 1e-10)
        losssum = PdQ - torch.log(PdQ + 1e-10)
        return losssum.mean()
