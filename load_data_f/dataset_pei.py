# from source import Source
import load_data_f.source as Source
from sklearn.datasets import load_digits
import torchvision.datasets as datasets
from sklearn.datasets import make_swiss_roll
from sklearn.preprocessing import StandardScaler
import numpy as np
import torch
from PIL import Image
import os
# import torchtext
import scanpy as sc
import scipy
from sklearn.decomposition import PCA
import pandas as pd


class PeiHumanTop2DataModule(Source.Source):

    def _LoadData(self):
        print('load DigitsDataModule')
        
        # if not self.train:
        #     random_state = self.random_state + 1
        datapei_raw = pd.read_csv('/root/data/PeiData/Human_diff.csv').iloc[:106]

        label_raw = datapei_raw['cell_type'].tolist()
        data = datapei_raw.drop(['cell_type'], axis=1).to_numpy()
        label_type = [d.split('_')[1] for d in label_raw]
        label_day = [float(d.split('_')[2][1:]) for d in label_raw]
        lab_all_undup = list(set(label_type))

        # digit = load_digits()
        self.data = torch.tensor(data).float()
        self.label = np.array([lab_all_undup.index(i) for i in label_type])
        self.inputdim = self.data[0].shape
        self.same_sigma = False
        print('shape = ', self.data.shape)
        self.label_str = [label_type, torch.tensor(label_day)]
        
class PeiHumanDataModule(Source.Source):

    def _LoadData(self):
        print('load DigitsDataModule')
        
        # if not self.train:
        #     random_state = self.random_state + 1
        datapei_raw = pd.read_csv('/root/data/PeiData/Human_diff.csv').iloc

        label_raw = datapei_raw['cell_type'].tolist()
        data = datapei_raw.drop(['cell_type'], axis=1).to_numpy()
        label_type = [d.split('_')[1] for d in label_raw]
        label_day = [float(d.split('_')[2][1:]) for d in label_raw]
        lab_all_undup = list(set(label_type))

        # digit = load_digits()
        self.data = torch.tensor(data).float()
        self.label = np.array([lab_all_undup.index(i) for i in label_type])
        self.inputdim = self.data[0].shape
        self.same_sigma = False
        print('shape = ', self.data.shape)
        self.label_str = [label_type, torch.tensor(label_day)]