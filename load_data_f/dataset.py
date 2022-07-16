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
from load_data_f.dataset_pei import *
from load_data_f.dataset_blod import *

class DigitsDataModule(Source.Source):

    def _LoadData(self):
        print('load DigitsDataModule')
        
        # if not self.train:
        #     random_state = self.random_state + 1

        digit = load_digits()
        self.data = torch.tensor(digit.data).float()
        self.label = digit.target
        self.inputdim = self.data[0].shape
        self.same_sigma = False
        print('shape = ', self.data.shape)
        self.label_str = [[ str(i) for i in list(self.label)]]
        
class MCADataModule(Source.Source):

    def _LoadData(self):

        print('load MCADataModule')
        # if not self.train:
        #     random_state = self.random_state + 1
        import pandas
        
        self.datapath = '/root/data/'

        data = np.load(self.datapath+'mca_data/mca_data_dim_34947.npy')
        label = np.load(self.datapath+'mca_data/mca_label_dim_34947.npy')
        
        data = PCA(n_components=50).fit_transform(data)
        # X_train, X_test, y_train, y_test = train_test_split(
        #     data, label, test_size=0.2, random_state=0)

        self.data = torch.tensor(data).float()
        self.label = torch.tensor(label)
        # self.data_test = torch.tensor(X_test).float()
        # self.label_test = torch.tensor(y_test)
        self.label_str = [[str(int(i)) for i in self.label]]
        self.inputdim = self.data[0].shape
        self.same_sigma = False
        print('shape = ', self.data.shape)
        
class MnistDataModule(Source.Source):

    def _LoadData(self):
        print('load MnistDataModule')
        
        dataloader = datasets.MNIST(
            root="/root/data", train=self.train, download=True, transform=None
        )
        
        self.data = dataloader.data[:self.args['n_point']].float() / 255
        self.label = dataloader.targets[:self.args['n_point']]
        self.label_str = [[ str(i) for i in list(dataloader.targets[:self.args['n_point']].numpy()) ]]
        self.inputdim = self.data[0].shape
        print('shape = ', self.data.shape)

class FMnistDataModule(Source.Source):

    def _LoadData(self):
        print('load MnistDataModule')
        
        dataloader = datasets.FashionMNIST(
            root="/root/data", train=self.train, download=True, transform=None
        )
        
        self.data = dataloader.data[:self.args['n_point']].float() / 255
        self.label = dataloader.targets[:self.args['n_point']]
        self.label_str = [[ str(i) for i in list(dataloader.targets[:self.args['n_point']].numpy()) ]]
        self.inputdim = self.data[0].shape
        print('shape = ', self.data.shape)

class M_handwrittenDataModule(Source.Source):
    # https://archive.ics.uci.edu/ml/datasets/Multiple+Features
    # This dataset consists of features of handwritten numerals (`0'--`9') extracted 
    # from a collection of Dutch utility maps. 200 patterns per class (for a total 
    # of 2,000 patterns) have been digitized in binary images. These digits are 
    # represented in terms of the following six feature sets (files):
    # 1. mfeat-fou: 76 Fourier coefficients of the character shapes;
    # 2. mfeat-fac: 216 profile correlations;
    # 3. mfeat-kar: 64 Karhunen-Love coefficients;
    # 4. mfeat-pix: 240 pixel averages in 2 x 3 windows;
    # 5. mfeat-zer: 47 Zernike moments;
    # 6. mfeat-mor: 6 morphological features.
    # In each file the 2000 patterns are stored in ASCI on 2000 lines. The first 200 
    # patterns are of class `0', followed by sets of 200 patterns for each of the 
    # classes `1' - `9'. Corresponding patterns in different feature sets (files) 
    # correspond to the same original character.


    def _LoadData(self):
        print('load M_handwrittenDataModule')
        
        datamat = scipy.io.loadmat('/root/data/MultiViewData/handwritten.mat')

        data = StandardScaler().fit_transform(
            np.concatenate([
                datamat['X'][0][0],
                datamat['X'][0][1],
                datamat['X'][0][2],
                datamat['X'][0][3],
                datamat['X'][0][4],
                datamat['X'][0][5],
                ], axis=1)
            )
        label = datamat['Y'].reshape((-1))

        self.data = torch.tensor(data).float()
        self.label = label
        self.label_str = [[ str(i) for i in self.label ]]
        self.inputdim = self.data[0].shape
        print('shape = ', self.data.shape)


# class IMDBDataModule(Source.Source):

#     def _LoadData(self):
#         print('load IMDBDataModule')
        
#         dataloader = torchtext.datasets.IMDB(
#             root="/root/data",
#         )
        
#         self.data = dataloader.data[:self.args['n_point']].float() / 255
#         self.label = dataloader.targets[:self.args['n_point']]
#         self.inputdim = self.data[0].shape
#         print('shape = ', self.data.shape)

class ActivityDataModule(Source.Source):
    def _LoadData(self):
        print('load ActivityDataModule')
        
        self.datapath = '/root/data/'
        # dataloader = datasets.KMNIST(
        #     root="~/data", train=self.train, download=True, transform=None
        # )
        # mat = scipy.io.loadmat(self.datapath+'feature_select/pixraw10P.mat')
        train_data = pd.read_csv(self.datapath+'feature_select/Activity_train.csv')
        test_data = pd.read_csv(self.datapath+'feature_select/Activity_test.csv')
        all_data = pd.concat([train_data, test_data])
        
        data = all_data.drop(['subject', 'Activity'], axis=1).to_numpy()
        label_str = all_data['Activity'].tolist()
        label_str_set = list(set(label_str))
        label = np.array([label_str_set.index(i) for i in label_str])

        data = (data-data.min())/(data.max()-data.min())
        # X_train, X_test, y_train, y_test = train_test_split(data, label, test_size=0.2, random_state=0)

        self.data = torch.tensor(data).float()
        self.label = torch.tensor(label)
        # self.data_test = torch.tensor(X_test).float()
        # self.label_test = torch.tensor(y_test)
        self.label_str = [[str(int(i)) for i in self.label]]
        self.inputdim = self.data[0].shape
        self.same_sigma = False
        print('shape = ', self.data.shape)


class EMnistDataModule(Source.Source):

    def _LoadData(self):
        print('load EMnistDataModule')
        
        dataloader = datasets.EMNIST(
            # root="/root/data", split='byclass', train=self.train, download=True, transform=None
            root="/root/data", split='balanced', train=self.train, download=True, transform=None
        )
        
        self.data = dataloader.data[:self.args['n_point']].float() / 255
        self.label = dataloader.targets[:self.args['n_point']]
        self.inputdim = self.data[0].shape
        self.label_str = [[ str(i) for i in list(dataloader.targets[:self.args['n_point']].numpy()) ]]
        print('shape = ', self.data.shape)

class EMnistBYCLASSDataModule(Source.Source):

    def _LoadData(self):
        print('load EMnistBYCLASSDataModule')
        
        printprint

        dataloader = datasets.EMNIST(
            # root="/root/data", split='byclass', train=self.train, download=True, transform=None
            root="/root/data", split='byclass', train=self.train, download=True, transform=None
        )
        
        self.data = dataloader.data[:self.args['n_point']].float() / 255
        print(self.data.shape)
        self.label = dataloader.targets[:self.args['n_point']]
        self.inputdim = self.data[0].shape
        self.label_str = [[ str(i) for i in list(dataloader.targets[:self.args['n_point']].numpy()) ]]
        print('shape = ', self.data.shape)

class EMnistBCDataModule(Source.Source):

    def _LoadData(self):
        print('load EMnistDataModule')
        
        dataloader = datasets.EMNIST(
            root="/root/data", split='byclass', train=self.train, download=True, transform=None
            # root="/root/data", split='balanced', train=self.train, download=True, transform=None
        )
        
        self.data = dataloader.data[:self.args['n_point']].float() / 255
        self.label = dataloader.targets[:self.args['n_point']]
        self.inputdim = self.data[0].shape
        self.label_str = [[ str(i) for i in list(dataloader.targets[:self.args['n_point']].numpy()) ]]
        print('shape = ', self.data.shape)

class SAMUSIKDataModule(Source.Source):

    def _LoadData(self):

        print('load SAMUSIKDataModule')
        # if not self.train:
        #     random_state = self.random_state + 1
        import pandas

        data = pandas.read_csv('/root/data/samusik_01.csv').to_numpy()[:,1:]
        lab_str = pandas.read_csv('/root/data/samusik01_labelnr.csv')['x'].to_list()
        # sadata_pca = sadata.X
        lab_all = list(set(lab_str))
        lab_all_undup = list(set(lab_all))
        label_train_numpy = np.array([lab_all.index(i) for i in lab_str])

        # input('-----------')
        self.data = torch.tensor(data,dtype=torch.float)
        self.label = np.array([lab_all_undup.index(i) for i in lab_str])
        self.label_str = [[str(i) for i in lab_str]]
        self.inputdim = self.data[0].shape
        self.same_sigma = False
        print('shape = ', self.data.shape)

class Cifar10DataModule(Source.Source):

    def _LoadData(self):
        print('load Cifar10DataModule')
        
        dataloader = datasets.CIFAR10(
            root="/root/data", train=self.train, download=True, transform=None
        )
        
        self.data = torch.tensor(dataloader.data[:self.args['n_point']]).float() / 255
        self.data = self.data.reshape((self.data.shape[0],-1))
        self.label = torch.tensor(dataloader.targets[:self.args['n_point']])
        self.inputdim = self.data[0].shape
        self.label_str = [[str(i) for i in self.label]]
        self.same_sigma = False
        print('shape = ', self.data.shape)

class Cifar100DataModule(Source.Source):

    def _LoadData(self):
        print('load Cifar10DataModule')
        
        dataloader = datasets.CIFAR100(
            root="/root/data", train=self.train, download=True, transform=None
        )
        
        self.data = torch.tensor(dataloader.data[:self.args['n_point']]).float() / 255
        self.data = self.data.reshape((self.data.shape[0],-1))
        self.label = torch.tensor(dataloader.targets[:self.args['n_point']])
        self.inputdim = self.data[0].shape
        self.label_str = [[str(i) for i in self.label]]
        self.same_sigma = False
        print('shape = ', self.data.shape)

class KMnistDataModule(Source.Source):

    def _LoadData(self):
        print('load KMnistDataModule')
        
        dataloader = datasets.KMNIST(
            root="/root/data", train=self.train, download=True, transform=None
        )
        
        self.data = dataloader.data[:self.args['n_point']].float() / 255
        self.label = dataloader.targets[:self.args['n_point']]
        self.inputdim = self.data[0].shape
        self.label_str = [[ str(i) for i in list(dataloader.targets[:self.args['n_point']].numpy()) ]]
        print('shape = ', self.data.shape)

class SwissRollDataModule(Source.Source):

    def _LoadData(self):
        print('load SwissRollDataModule')
        
        data_ = make_swiss_roll(
            n_samples=2000,
            noise=0.0, 
            random_state=1
        )
        
        data = torch.tensor(data_[0]/50).float()
        label = data_[1]
        print(data.shape)
        
        
        self.data = torch.tensor(data).float()
        self.label = label
        self.inputdim = self.data[0].shape
        self.same_sigma = False
        self.label_str = [label]

class SwissRoll2DataModule(Source.Source):

    def _LoadData(self):
        print('load SwissRollDataModule')
        swiss_roll1 = make_swiss_roll(
            n_samples=2000,
            noise=0.0, 
            random_state=1
        )
        swiss_roll2 = make_swiss_roll(
            n_samples=2000,
            noise=0.0, 
            random_state=1
        )
        data1 = swiss_roll1[0] /50
        data2 = swiss_roll2[0] /50
        label1 = swiss_roll1[1]
        label2 = swiss_roll2[1]
        data1[:,1] = data1[:,1]/2
        data2[:,1] = data2[:,1]/2

        data2 = np.concatenate([data2[:,1:2],data2[:,0:1],data2[:,2:3]], axis=1)

        data2 = data2*np.array([1, 1, -1])
        data2 = data2+np.array([-0.07, 0.1, -0.35])
        data = np.concatenate([data1,data2])
        label = np.concatenate([label1,label2])
        print(data.shape)
        
        
        self.data = torch.tensor(data).float()
        self.label = label
        self.inputdim = self.data[0].shape
        self.same_sigma = False
        self.label_str = [label]


class Coil20DataModule(Source.Source):

    def _LoadData(self):
        print('load Coil20DataModule')
        
        path = "/root/data/coil-20-proc"
        fig_path = os.listdir(path)
        fig_path.sort()

        label = []
        data = np.zeros((1440, 128, 128))
        for i in range(1440):
            I = Image.open(path + "/" + fig_path[i])
            I_array = np.array(I)
            data[i] = I_array
            label.append(int(fig_path[i].split("__")[0].split("obj")[1]))
        
        self.data = torch.tensor(data / 255).float()
        self.label = label
        self.inputdim = self.data[0].shape
        self.same_sigma = False
        self.label_str = [[str(i) for i in label]]
        
class Coil100DataModule(Source.Source):

    def _LoadData(self):
        print('load Coil20DataModule')
        
        path = "/root/data/coil-100"
        fig_path = os.listdir(path)

        label = []
        data = np.zeros((100 * 72, 128, 128, 3))
        for i, path_i in enumerate(fig_path):
            # print(i)
            if "obj" in path_i:
                I = Image.open(path + "/" + path_i)
                I_array = np.array(I.resize((128, 128)))
                data[i] = I_array
                label.append(int(fig_path[i].split("__")[0].split("obj")[1]))

        self.data = torch.tensor(np.swapaxes(data, 1, 3) / 255).float()
        self.label = np.array(label)
        self.inputdim = self.data[0].shape
        self.label_str = [[str(i) for i in label]]
        self.same_sigma = False
        print('shape = ', self.data.shape)
        
class SmileDataModule(Source.Source):

    def _LoadData(self):

        print('load SmileDataModule')
        # if not self.train:
        #     random_state = self.random_state + 1
        adata=np.load('data/smile01.npy')
        index = np.arange(adata.shape[0])
        np.random.shuffle(index)
        adata = adata[index[:3000]]
        print(adata.shape)
        
                
        self.data = torch.tensor(adata[:,0:2]).float()
        self.label = adata[:,2]
        self.inputdim = self.data[0].shape
        self.same_sigma = False
        self.label_str = [[str(i) for i in self.label ]]
        print('shape = ', self.data.shape)



        
class ToyDiffDataModule(Source.Source):

    def _LoadData(self):
        print('load ToyDiffDataModule')

        np.random.RandomState(1)

        n = (900//3)
        data0 = np.array([0.0]*100) + np.random.randn(n, 100)/3
        data5 = [2]*100 + np.random.randn(n, 100)/2
        data10 = [4.0]*100 + np.random.randn(n, 100)/1
        
        data = []
        label = []
        for i in range(n):
            data.append(data0[i])
            data.append(data5[i])
            data.append(data10[i])
            label.append(0)
            label.append(1)
            label.append(2)
                
        self.data = torch.tensor(data).float()
        self.label = label
        self.inputdim = self.data[0].shape
        self.same_sigma = True
        
        
class SeversphereDataModule(Source.Source):

    def _LoadData(self):

        print('load SeversphereDataModule')
        # if not self.train:
        #     random_state = self.random_state + 1
        from sklearn.utils import check_random_state
        
        # n_samples = 2000
        # random_state = check_random_state(0)
        # p = random_state.rand(n_samples) * (2 * np.pi - 0.55)
        # t = random_state.rand(n_samples) * np.pi

        # # Sever the poles from the Seversphere.
        # indices = ((t < (np.pi - (np.pi / 8))) & (t > ((np.pi / 8))))
        # colors = p[indices]
        # x, y, z = np.sin(t[indices]) * np.cos(p[indices]), \
        #     np.sin(t[indices]) * np.sin(p[indices]), \
        #     np.cos(t[indices])
        n_samples = 10000
        random_state = check_random_state(0)
        p = random_state.rand(n_samples) * (2 * np.pi - 0.005)
        t = random_state.rand(n_samples) * np.pi

        # Sever the poles from the sphere.


        indices = ((t < (np.pi - (np.pi * 2/ 8))) & (t > ((np.pi * 2/ 8))))
        r = 1 + np.sin(p[indices]*5)/8
        colors = p[indices]
        x, y, z = r * np.sin(t[indices]) * np.cos(p[indices]), \
                r * np.sin(t[indices]) * np.sin(p[indices]), \
                np.cos(t[indices])


        data_train = torch.tensor(
            np.concatenate(
                [x[:,None], y[:,None], z[:,None]],
                axis=1
                )
            ).float()
        label_train = colors
                
        self.data = data_train
        self.label = label_train
        self.inputdim = self.data[0].shape
        self.same_sigma = False
        self.label_str = [self.label]
        print('shape = ', self.data.shape)        
        
        
class ColonDataModule(Source.Source):

    def _LoadData(self):

        print('load ColonDataModule')
        # if not self.train:
        #     random_state = self.random_state + 1
        sadata = sc.read("/root/data/colonn.h5ad")
        # sadata = sc.read('/usr/data/mldl-elis/ELIS_evalu_new/data/colonpca.h5ad') 
        # sadata_pca = sadata.obsm['X_pca'] 
        sadata_pca = sadata.X
        sadata_pca = torch.tensor(sadata_pca)
        data_train = sadata_pca
        lab = np.array([int(i) for i in list(sadata.obs.clusterstr)])
        label_train = torch.tensor(lab)  
        
                
        self.data = data_train
        self.label = label_train
        self.inputdim = self.data[0].shape
        self.same_sigma = False
        self.label_str = [[str(i) for i in self.label]]
        print('shape = ', self.data.shape)

class Gast10kDataModule(Source.Source):

    def _LoadData(self):

        print('load Gast10kDataModule')
        # if not self.train:
        #     random_state = self.random_state + 1
        sadata = sc.read("/root/data/gast10kwithcelltype.h5ad")
        # sadata = sc.read('/usr/data/mldl-elis/ELIS_evalu_new/data/colonpca.h5ad') 
        # sadata_pca = sadata.obsm['X_pca'] 
        sadata_pca = sadata.obsm['X_pca']
        data_train = torch.tensor(sadata_pca)
 
        label_train_str = list(sadata.obs['celltype'])
        label_train_str_set = list(set(label_train_str))
        label_train = torch.tensor([label_train_str_set.index(i) for i in label_train_str])

        label_train_str_2 = list(sadata.obs['hist'])
        # label_train_str_set_2 = list(set(label_train_str_2))
        # label_train_2 = torch.tensor([label_train_str_set_2.index(i) for i in label_train_str_2])
                
        self.data = data_train
        self.label = label_train
        self.label_str = [np.array(label_train_str), np.array(label_train_str_2)]
        self.inputdim = self.data[0].shape
        self.same_sigma = False
        print('shape = ', self.data.shape)
        
        
class HCL60KDataModule(Source.Source):

    def _LoadData(self):

        print('load HCL60KDataModule')
        # if not self.train:
        #     random_state = self.random_state + 1
        sadata = sc.read("/root/data/HCL60kafter-elis-all.h5ad")
        # sadata = sc.read('/usr/data/mldl-elis/ELIS_evalu_new/data/colonpca.h5ad') 
        # sadata_pca = sadata.obsm['X_pca'] 
        sadata_pca = sadata.obsm['X_pca']
        sadata_pca = torch.tensor(sadata_pca)
        data_train = sadata_pca
        lab = np.array([int(i) for i in list(sadata.obs.louvain)])
        label_train = torch.tensor(lab)  
        
        self.data = data_train
        self.label = label_train
        self.label_str = [[ str(i) for i in list(self.label)]]
        self.inputdim = self.data[0].shape
        self.same_sigma = False
        print('shape = ', self.data.shape)
        
class HCL280KDataModule(Source.Source):

    def _LoadData(self):

        print('load HCL280KDataModule')
        # from sklearn.decomposition import PCA
        # if not self.train:
        #     random_state = self.random_state + 1
        # sadata = sc.read("/root/data/HCL_Fig1_adata_general_model_celltype_no_process_batch_train_data.h5ad")
        # sadata_pca = sadata.X
        # sadata_pca = torch.tensor(PCA(n_components=50).fit_transform(sadata_pca))
        # data_train = sadata_pca
        sadata_pca = torch.tensor(np.load('/root/data/hcl_data_dim_50.npy'))
        label_train_str = np.load('/root/data/hcl_label_dim_50.npy')
        # lab = np.array([int(i) for i in list(sadata.obs.louvain)])
        # label_train_str = list(sadata.obs['celltype'])
        # np.save('HCL60Kdata.npy', label_train_str)
        lab_all = list(set(label_train_str))
        label_train_numpy = np.array([lab_all.index(i) for i in label_train_str])
        # input('-----------')
        self.data = sadata_pca
        self.label = label_train_numpy
        self.label_str = [[str(i) for i in label_train_str]]
        self.inputdim = self.data[0].shape
        self.same_sigma = False
        print('shape = ', self.data.shape)


class PBMCDataModule(Source.Source):

    def _LoadData(self):

        print('load PBMCDataModule')
        # if not self.train:
        #     random_state = self.random_state + 1
        adata = sc.read("/root/data/PBMC3k_HVG_regscale.h5ad")
        data_train = torch.tensor(adata.obsm['X_pca'].copy())
        label_train_str = list(adata.obs['celltype'])
        label_train_str_set = list(set(label_train_str))
        label_train = torch.tensor([label_train_str_set.index(i) for i in label_train_str])

        self.data = data_train
        self.label = label_train
        self.label_str = [np.array(label_train_str)]
        self.inputdim = self.data[0].shape
        self.same_sigma = False
        print('shape = ', self.data.shape)