import os
from multiprocessing import Process, Manager
import numpy as np
import signal
import time
from itertools import product
import subprocess

# parameter analysis for SAGloss

import tool

n_point_list = [60000]
data_name_list = ['Coil20']
# data_name_list = ['EMnist']
perplexity_list = [10,20,40,60,]
# lr_list = [1e-4, 1e-4, 2e-5, 1e-5, 5e-5,]
lr_list = [1e-4, 1e-3, 1e-2,]
batch_size_list = [1440, ]
vs_list = [1e-3,]
ve_list = [-1]
method_list = ['dmt']
K_list = [2, 5, 10, 20, 40]
num_latent_dim_list = [2]
augNearRate_list = [10, 1000, 100000, 10000000]

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
]

mainFunc = "./main_pl_origin.py"
ater = tool.AutoTrainer(
    changeList,
    paramName,
    mainFunc,
    deviceList=cudalist,
    poolNumber=3*len(cudalist),
    name="autotrain",
    waittime=3,
)
ater.Run()
