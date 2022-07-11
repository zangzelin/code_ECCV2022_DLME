import os
from multiprocessing import Process, Manager
import numpy as np
import signal
import time
from itertools import product
import subprocess

# parameter analysis for SAGloss

import tool

n_point_list = [600000]
data_name_list = ['Seversphere']
perplexity_list = [20,]
lr_list = [1e-3, 1e-2, 5e-3]
batch_size_list = [300, 800, 1000]
# vs_list = [1e-3, 5e-3, 1e-2, 5e-2, 1e-1, 5e-1]
vs_list = [1e-4, 5e-4, 1e-3, 5e-3, 1e-2, 5e-2]
ve_list = [-1]
method_list = ['dmt']
K_list = [3, 5, 10, 15]
num_latent_dim_list = [3]
augNearRate_list = [10, 100, 1000, 10000]
epochs_list = [15000]


cudalist = [
    0,
    1,
    2,
    3,
    4,
    5,
    6,
    7,
]

changeList = [
    n_point_list,
    data_name_list,
    perplexity_list,
    batch_size_list,
    lr_list,
    vs_list,
    ve_list,
    method_list,
    K_list,
    num_latent_dim_list,
    augNearRate_list,
    epochs_list,
    ]

paramName = [
    'n_point',
    'data_name',
    'perplexity',
    'batch_size',
    'lr',
    'vs',
    've',
    'method',
    'K',
    'num_latent_dim',
    'augNearRate',
    'epochs',
]

mainFunc = "./main.py"
ater = tool.AutoTrainer(
    changeList,
    paramName,
    mainFunc,
    deviceList=cudalist,
    poolNumber=2*len(cudalist),
    name="autotrain",
    waittime=1,
)
ater.Run()


