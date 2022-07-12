source /zangzelin/.bashrc; 
# <<< conda initialize <<<

cd /zangzelin/project/scl/
pwd

if [ -f "/opt/anaconda3/etc/profile.d/conda.sh" ]; then
    . "/opt/anaconda3/etc/profile.d/conda.sh"
    CONDA_CHANGEPS1=false conda activate torch1.8
fi


conda-env list

export http_proxy=http://192.168.105.204:3128
export https_proxy=http://192.168.105.204:3128

export LC_ALL=C.UTF-8
/opt/anaconda3/envs/torch1.8/bin/python -m pip install wandb -U


wandb login --host=http://172.16.55.92:1080 local-82bf782e68129f9b30afc15675d7bf1a3b041c1c


# bash ./bash/wandb.sh
CUDA_VISIBLE_DEVICES=0 bash ./bash/wandb.sh &
CUDA_VISIBLE_DEVICES=1 bash ./bash/wandb.sh &
# CUDA_VISIBLE_DEVICES=0 bash ./bash/wandb.sh &
# CUDA_VISIBLE_DEVICES=1 bash ./bash/wandb.sh &
CUDA_VISIBLE_DEVICES=0 bash ./bash/wandb.sh &
CUDA_VISIBLE_DEVICES=1 bash ./bash/wandb.sh 


# CUDA_VISIBLE_DEVICES=1 wandb agent zangzelin/PatEmb/fxmn1939 &
# CUDA_VISIBLE_DEVICES=2 wandb agent zangzelin/PatEmb/fxmn1939 &
# sleep 30s
# CUDA_VISIBLE_DEVICES=3 wandb agent zangzelin/PatEmb/fxmn1939 

/opt/anaconda3/envs/torch1.8/bin/python /zangzelin/project/otn/bash/check_mem.py

sleep 600m

sleep 30m
