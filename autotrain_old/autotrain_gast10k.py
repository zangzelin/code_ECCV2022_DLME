import os
from multiprocessing import Process, Manager
import numpy as np
import signal
import time
from itertools import product
import subprocess

# parameter analysis for SAGloss

import tool

data_name_list = ['Gast10k']
perplexity_list = [10, 20, 40]
lr_list = [ 1e-4, 1e-3, 1e-2]
vs_list = [1e-5, 1e-4, 1e-3, 1e-2, 1e-1]
ve_list = [-1, 1e-3, 1e-2, 1e-1, 1,]
method_list = ['dmt_mask']
batch_size_list = [3000]

cudalist = [
    0,
    # 1,
    2,
    3,
    4,
    5,
    6,
    7,
]

changeList = [
    batch_size_list,
    data_name_list,
    perplexity_list,
    lr_list,
    vs_list,
    ve_list,
    method_list,
    ]

paramName = [
    'batch_size',
    'data_name',
    'perplexity',
    'lr',
    'vs',
    've',
    'method',
]

mainFunc = "./main_pl_origin.py"
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
