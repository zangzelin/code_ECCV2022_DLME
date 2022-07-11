import os
from multiprocessing import Process, Manager
import numpy as np
import signal
import time
from itertools import product
import subprocess

# parameter analysis for SAGloss

import tool

data_name_list = ['Cifar10']
perplexity_list = [10, 15, 20]
lr_list = [1e-4]
vs_list = [1e-3, 1e-2, 1e-1]
ve_list = [-1]
method_list = ['dmt_mask']
augNearRate_list = [10, 100, 1000]
K_list = [5, 10, 15, 20]

cudalist = [
    # 0,
    # 1,
    2,
    3,
    4,
    5,
    6,
    7,
    8,
    9,
]

changeList = [
    data_name_list,
    perplexity_list,
    lr_list,
    vs_list,
    ve_list,
    method_list,
    augNearRate_list,
    K_list,
    ]

paramName = [
    'data_name',
    'perplexity',
    'lr',
    'vs',
    've',
    'method',
    'augNearRate',
    'K',
]

mainFunc = "./main_pl_aug.py"
ater = tool.AutoTrainer(
    changeList,
    paramName,
    mainFunc,
    deviceList=cudalist,
    poolNumber=4*len(cudalist),
    name="autotrain",
    waittime=1,
)
ater.Run()
