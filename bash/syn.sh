source /zangzelin/.bashrc; 
# <<< conda initialize <<<

cd /zangzelin/project/otn/
pwd

if [ -f "/opt/anaconda3/etc/profile.d/conda.sh" ]; then
    . "/opt/anaconda3/etc/profile.d/conda.sh"
    CONDA_CHANGEPS1=false conda activate torch1.8
fi


conda-env list

export http_proxy=http://192.168.105.204:3128
export https_proxy=http://192.168.105.204:3128

export LC_ALL=C.UTF-8
/opt/anaconda3/envs/torch1.8/bin/python -m pip install wandb

wandb sync wandb/offline-run-2021112*