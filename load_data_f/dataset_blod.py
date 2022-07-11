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


class BlodZHEERDataModule(Source.Source):

    def _LoadData(self):
        # print('load DigitsDataModule')
        
        # if not self.train:
        #     random_state = self.random_state + 1
        datapei_raw = pd.read_excel('/root/data/bold/all_hospital_v1_v2_for_article.xls')
        datapei_raw = datapei_raw.replace('N', 0)
        datapei_raw = datapei_raw.replace('Y', 1)

        datapei_raw = datapei_raw[
            datapei_raw['hospital'].str.contains('zheer')
            ]
        labelname = ['d'+str(i+1) for i in range(14)]
        label = np.log(1+datapei_raw[labelname].sum(axis=1).to_numpy())
        
        data = datapei_raw.drop(['blood_type','hospital']+labelname, axis=1)
        data = data.fillna(data.median()).to_numpy()

        # label_raw = datapei_raw['cell_type'].tolist()
        # data = datapei_raw.drop(['cell_type'], axis=1).to_numpy()
        # label_type = [d.split('_')[1] for d in label_raw]
        # label_day = [float(d.split('_')[2][1:]) for d in label_raw]
        # lab_all_undup = list(set(label_type))

        # digit = load_digits()
        self.data = torch.tensor(data).float()
        self.label = np.array(label)
        self.inputdim = self.data[0].shape
        self.same_sigma = False
        print('shape = ', self.data.shape)
        self.label_str = [label]


class BlodAllDataModule(Source.Source):

    def _LoadData(self):
        # print('load DigitsDataModule')
        
        # if not self.train:
        #     random_state = self.random_state + 1
        datapei_raw = pd.read_excel('/root/data/bold/all_hospital_v1_v2_for_article.xls')
        datapei_raw = datapei_raw.replace('N', 0)
        datapei_raw = datapei_raw.replace('N ', 0)
        datapei_raw = datapei_raw.replace('Y', 1)
        datapei_raw = datapei_raw.replace('Y ', 1)
        datapei_raw = datapei_raw.replace('y', 1)
        datapei_raw = datapei_raw.replace('B', 0)

        # datapei_raw = datapei_raw[
        #     datapei_raw['hospital'].str.contains('zheer')
        #     ]
        labelname = ['d'+str(i+1) for i in range(14)]
        label = np.log(1+datapei_raw[labelname].sum(axis=1).to_numpy())
        
        data = datapei_raw.drop(['blood_type','hospital']+labelname, axis=1)
        data = data.fillna(data.median()).to_numpy().astype(np.float32)

        # label_raw = datapei_raw['cell_type'].tolist()
        # data = datapei_raw.drop(['cell_type'], axis=1).to_numpy()
        # label_type = [d.split('_')[1] for d in label_raw]
        # label_day = [float(d.split('_')[2][1:]) for d in label_raw]
        # lab_all_undup = list(set(label_type))

        print(data)

        # digit = load_digits()
        self.data = torch.tensor(data).float()
        self.label = np.array(label)
        self.inputdim = self.data[0].shape
        self.same_sigma = False
        print('shape = ', self.data.shape)
        self.label_str = [label]
    
# class PeiHumanDataModule(Source.Source):

#     def _LoadData(self):
#         print('load DigitsDataModule')
        
#         # if not self.train:
#         #     random_state = self.random_state + 1
#         datapei_raw = pd.read_csv('/root/data/PeiData/Human_diff.csv').iloc

#         label_raw = datapei_raw['cell_type'].tolist()
#         data = datapei_raw.drop(['cell_type'], axis=1).to_numpy()
#         label_type = [d.split('_')[1] for d in label_raw]
#         label_day = [float(d.split('_')[2][1:]) for d in label_raw]
#         lab_all_undup = list(set(label_type))

#         # digit = load_digits()
#         self.data = torch.tensor(data).float()
#         self.label = np.array([lab_all_undup.index(i) for i in label_type])
#         self.inputdim = self.data[0].shape
#         self.same_sigma = False
#         print('shape = ', self.data.shape)
#         self.label_str = [label_type, torch.tensor(label_day)]