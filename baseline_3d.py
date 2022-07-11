import os
import sys

import numpy as np
import sklearn
import torch
from PIL import Image
from sklearn.datasets import fetch_openml, make_s_curve, make_swiss_roll

import wandb
import umap
import load_data_f.dataset as datasetfunc
import pytorch_lightning as pl
import load_disF.disfunc as disfunc
import load_simF.simfunc as simfunc
import eval.eval_core as eval_core
import plotly.express as px
from sklearn.manifold import TSNE
from umap.parametric_umap import ParametricUMAP
from ivis import Ivis
from sklearn.preprocessing import MinMaxScaler
import phate
import networkx as nx
import random

import tool
if __name__ == '__main__':

    
    import argparse
    parser = argparse.ArgumentParser(description='*** author')
    parser.add_argument('--name', type=str, default='digits_T', )
    parser.add_argument('--data_name', type=str, default='EMnist', 
                        choices=[
                            'Digits', 'Coil20', 'Coil100',
                            'Smile', 'ToyDiff', 'SwissRoll',
                            'KMnist', 'EMnist', 'Mnist',
                            'EMnistBC', 'EMnistBYCLASS',
                            'Cifar10', 'Colon',
                            'Gast10k', 'HCL60K', 'PBMC', 
                            'HCL280K', 'SAMUSIK', 
                            'M_handwritten', 'Seversphere'
                            ])
    parser.add_argument('--n_point', type=int, default=600000000, )
    parser.add_argument('--perplexity', type=int, default=20)
    parser.add_argument('--v_input', type=float, default=100)
    parser.add_argument('--metric', type=str, default="euclidean", )
    parser.add_argument('--pow_input', type=float, default=2)
    parser.add_argument('--same_sigma', type=bool, default=False)
    parser.add_argument('--K', type=int, default=5)
    parser.add_argument('--method', type=str, default='umap', 
                        choices=['tsne', 'umap','pumap', 'ivis', 'phate'])
    args = pl.Trainer.add_argparse_args(parser)
    args = args.parse_args()
    
    disfunc_use = getattr(disfunc, 'EuclideanDistanceNumpy')
    simfunc_use = getattr(simfunc, 'UMAPSimilarity')
    simfunc_npuse = getattr(simfunc, 'UMAPSimilarityNumpy')
    dm_class = getattr(datasetfunc, args.__dict__['data_name'] + 'DataModule')
    
    runname = 'baseline_{}_{}_{}'.format(
        args.__dict__['data_name'], 
        args.__dict__['method'], 
        args.__dict__['metric'])
    wandb.init(
        name=runname,
        project='DLME_manifold_BL3D',
        entity='zangzelin',
        # offline=False,
    )
    dataset = dm_class(
        DistanceF=disfunc_use,
        SimilarityF=simfunc_use,
        SimilarityNPF=simfunc_npuse,
        jumpPretreatment=True,
        **args.__dict__,
        )
    data = dataset.data.detach().cpu().numpy().reshape(dataset.data.shape[0], -1)
    if args.__dict__['data_name'] == 'Colon':
        labelstr = np.array(dataset.label_str[0])
    else:
        labelstr = np.array(dataset.label_str)

    label = np.array(dataset.label)
    
    # e = eval_core.Eval(
    #     input=data,
    #     latent=data,
    #     label=label,
    #     k=10
    #     )
    # e.E_Curance("SwissRoll" in args.__dict__['data_name'])
    
    if args.__dict__['method'] == 'umap':
        latent = umap.UMAP(n_components=3, random_state=0).fit_transform(data)
    if args.__dict__['method'] == 'tsne':
        latent = TSNE(n_components=3, random_state=0).fit_transform(data)
    if args.__dict__['method'] == 'pumap':
        embedder = ParametricUMAP(dims=3, random_state=0)
        latent = embedder.fit_transform(data)
    if args.__dict__['method'] == 'ivis':
        X_scaled = MinMaxScaler().fit_transform(data)
        model = Ivis(embedding_dims=3, )
        latent = model.fit_transform(X_scaled)
    if args.__dict__['method'] == 'phate':
        phate_op = phate.PHATE(n_components=3, random_state=0)
        latent = phate_op.fit_transform(data)
    
    print('----------------')
    print('----------------')
    print(args.__dict__['method'])
    print('----------------')
    print('----------------')
    
    e = eval_core.Eval(
        input=data,
        latent=latent,
        label=label,
        k=10
        )

    # if 'SwissRoll' in args.__dict__['data_name']:
    #     label_root = np.array([0]*data.shape[0])
    # else:
    #     label_root = label

    # path_list = Curance_path_list(neighbour_input=e.neighbour_input, distance_input=e.distance_input)
    result_dict = {
        # 'epoch': self.current_epoch,
        # 'metric/Mrremean': np.mean(e.E_mrre()),
        # 'metric/Continuity': e.E_continuity(),
        # 'metric/Trustworthiness':e.E_trustworthiness(),
        # 'metric/Pearson': e.E_Rscore(),
        'SVC':e.E_Classifacation_SVC(),
        'Curance':e.E_Curance("SwissRoll" in args.__dict__['data_name']),
        'Curance_2':e.E_Curance_2("SwissRoll" in args.__dict__['data_name']),
        'Kmeans':e.E_Clasting_Kmeans(),
        # 'metric/Dismatcher':e.E_Dismatcher(),
        # 'ACCKNN':e.E_Classifacation_KNN(),
        'visualize/embdeing0': px.scatter_3d(
                    x=latent[:,0],
                    y=latent[:,1],
                    z=latent[:,2],
                    color=np.array(dataset.label_str[0])
            )
        }
    
    if args.__dict__['data_name'] != 'SwissRoll':
        result_dict['metric/Dismatcher'] = e.E_Dismatcher(),
    g = tool.GIFPloter()
    g.AddNewFig(
        latent=latent,
        label=label,
        title_=args.__dict__['method']+args.__dict__['data_name'],
        path='baseline_figout/',
        )

    # result_dict['visualize/embdeing'] = px.scatter(x=latent[:,0], y=latent[:,1], color=labelstr[0])
    wandb.log(result_dict)