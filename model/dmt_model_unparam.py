# a pytorch based lisv2 code
# author: ***
# email: ***@gmail.com

import functools
import pdb
import time
from locale import currency
from multiprocessing import Pool
from typing import Any

import numpy as np
import torch
import torch.autograd
from torch import nn, set_flush_denormal
# import  as RestNet
from model.LeNet import MyLeNet
from model.ResNet_ import resnet18
from sklearn.manifold import SpectralEmbedding
from sklearn.manifold import LocallyLinearEmbedding

class LISV2_Model(torch.nn.Module):
    def __init__(
        self,
        input_dim: list,
        device: Any,
        NetworkStructure: list,
        dataset_name: str,
        model_type: str,
        latent_dim=2,
        num_point=1000,
        input_data=None,
    ):

        super(LISV2_Model, self).__init__()
        with torch.no_grad():
            self.device = device
            self.phis = []
            self.n_dim = latent_dim
            self.NetworkStructure = NetworkStructure
            self.NetworkStructure[0] = functools.reduce(lambda x,y:x * y, input_dim)
            self.dataset_name = dataset_name
            self.model_type = model_type

            # self.fc = nn.Linear( latent_dim, num_point, bias=False)

            # init_emb = LocallyLinearEmbedding(n_components=2).fit_transform(
            #     input_data.reshape(input_data.shape[0], -1))
            init_emb = np.random.randn(num_point,latent_dim)
            self.emb = nn.Parameter(
                torch.tensor(
                    init_emb / (np.std(init_emb)/np.std(input_data.detach().cpu().numpy()))
                    )
            )
    
    # def Init_embedding(self, emb):
            # self.emb.weight.data = 
            # self.fc.weight.data = torch.randn(num_point, latent_dim)
            # self.fc.weight = self.fc.weight 
        

    def forward(self, data, index=None):

        x = self.emb[index]
        # x = x/torch.std(x)
        return x



