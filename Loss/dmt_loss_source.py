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
import sklearn


class Source(nn.Module):
    def __init__(
        self,
        v_input,
        SimilarityFunc,
        metric="braycurtis",
    ):
        super(Source, self).__init__()

        self.v_input = v_input
        self.gamma_input = self._CalGamma(v_input)
        self.ITEM_loss = self._TwowaydivergenceLoss
        self._Similarity = SimilarityFunc
        self.metric = metric
    
    def forward(self, input_data, latent_data, rho, sigma, v_latent):
        
        dis_P = self._DistanceSquared(input_data)
        P = self._Similarity(
                dist=dis_P,
                rho=rho,
                sigma_array=sigma,
                gamma=self.gamma_input,
                v=self.v_input)
        dis_Q = self._DistanceSquared(latent_data)
        Q = self._Similarity(
                dist=dis_Q,
                rho=0,
                sigma_array=1,
                gamma=self._CalGamma(v_latent),
                v=v_latent)
        
        self.Psave = P.detach().cpu().numpy()
        self.Qsave = Q.detach().cpu().numpy()
        self.dis_Qsave = dis_Q.detach().cpu().numpy()
        loss_ce = self.ITEM_loss(P=P, Q=Q)
        
        return loss_ce

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

    def _L2Loss(self, P, Q):

        losssum = torch.norm(P-Q, p=2)/P.shape[0]
        return losssum
    
    def _L3Loss(self, P, Q):

        losssum = torch.norm(P-Q, p=3)/P.shape[0]
        return losssum
    
    def _DistanceSquared(
        self,
        x,
        metric='euclidean'
    ):
        if metric == 'euclidean':
            m, n = x.size(0), x.size(0)
            xx = torch.pow(x, 2).sum(1, keepdim=True).expand(m, n)
            yy = xx.t()
            dist = xx + yy
            dist.addmm_(1, -2, x, x.t())
            dist = dist.clamp(min=1e-12)
        elif metric == 'braycurtis':
            # import sklearn.metrics.pairwise_distances as pairwise_distances
            dist = torch.tensor(
                sklearn.metrics.pairwise_distances(
                    X=x.detach().cpu().numpy(),
                    Y=x.detach().cpu().numpy(),
                    metric='braycurtis',
                    n_jobs=-1,
                    )
                ).cuda()
        # d[torch.eye(d.shape[0]) == 1] = 1e-12

        return dist

    def _CalGamma(self, v):
        
        a = scipy.special.gamma((v + 1) / 2)
        b = np.sqrt(v * np.pi) * scipy.special.gamma(v / 2)
        out = a / b

        return out