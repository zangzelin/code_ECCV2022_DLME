import os
from multiprocessing import Process,Manager
import numpy as np
import signal
import time
from itertools import product
import subprocess

# parameter analysis for SAGloss

import tool

n_point_list = [600000]
data_name_list = [
    # 'SwissRoll',
    # 'Seversphere',
    # 'Digits',
    # 'Coil20',
    # 'Coil100',
    # 'KMnist',
    # 'EMnist',
    # 'Mnist',
    # 'EMnistBC',
    # 'EMnistBYCLASS',
    # 'Colon',
    # 'Gast10k',
    # 'HCL60K',
    'PBMC',
    # 'HCL280K',
    # 'SAMUSIK',
    ]
method_list = [
    # 'tsne',
    # 'umap',
    # 'pumap',
    # 'ivis',
    'phate',
    ]


cudalist = [
    0,
    1,
    # 2,
    # 3,
    # 4,
    # 5,
    # 6,
    # 7,
]

changeList = [
    n_point_list,
    data_name_list,
    method_list,
    ]

paramName = [
    'n_point',
    'data_name',
    'method',
]

mainFunc = "./baseline_3d.py"
ater = tool.AutoTrainer(
    changeList,
    paramName,
    mainFunc,
    deviceList=cudalist,
    poolNumber=1*len(cudalist),
    name="autotrain",
    waittime=1,
)
ater.Run()


