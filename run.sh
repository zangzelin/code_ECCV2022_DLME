# python main_pl_origin.py --method=dmt --data_name=Smile --eta=0 --lr=0.0001 --perplexity=15 --pow_input=2 --pow_latent=2 --ve=100 --vs=0.001
# python main_pl_origin.py --method=dmt_mask --data_name=Smile --eta=0 --lr=0.0001 --perplexity=15 --pow_input=2 --pow_latent=2 --ve=100 --vs=0.001

# python main_pl_origin.py --method=dmt --data_name=Digits --eta=0 --lr=0.0005 --perplexity=20 --pow_input=2 --pow_latent=2 --ve=-1 --vs=0.01
# python main_pl_origin.py --method=dmt_mask --data_name=Digits --eta=0 --lr=0.0005 --perplexity=20 --pow_input=2 --pow_latent=2 --ve=-1 --vs=0.01
# nohup python main_pl_origin.py --method=dmt --data_name=Coil100 --eta=0 --lr=0.0001 --perplexity=15 --pow_input=2 --pow_latent=2 --ve=100 --vs=0.001 &
# nohup python main_pl_origin.py --method=dmt_mask --data_name=Coil100 --eta=0 --lr=0.0001 --perplexity=15 --pow_input=2 --pow_latent=2 --ve=100 --vs=0.001 &

# CUDA_VISIBLE_DEVICES=1 nohup python main_pl_origin.py --method=dmt --data_name=Mnist --eta=0 --lr=0.0001 --perplexity=15 --pow_input=2 --pow_latent=2 --ve=-1 --vs=0.001 &
# CUDA_VISIBLE_DEVICES=1 nohup python main_pl_origin.py --method=dmt_mask --data_name=Mnist --eta=0 --lr=0.0001 --perplexity=15 --pow_input=2 --pow_latent=2 --ve=-1 --vs=0.001 &

# nohup python main_pl_origin.py --method=dmt --data_name=Coil20 --eta=0 --lr=0.001 --perplexity=10 --pow_input=2 --pow_latent=2 --ve=0.05 --vs=0.01 &
# nohup python main_pl_origin.py --method=dmt_mask --data_name=Coil20 --eta=0 --lr=0.001 --perplexity=10 --pow_input=2 --pow_latent=2 --ve=0.05 --vs=0.01 &

# nohup python main_pl_origin.py --method=dmt --data_name=ToyDiff --eta=0 --lr=0.0001 --perplexity=15 --pow_input=2 --pow_latent=2 --ve=100 --vs=0.001 &
# nohup python main_pl_origin.py --method=dmt_mask --data_name=ToyDiff --eta=0 --lr=0.0001 --perplexity=15 --pow_input=2 --pow_latent=2 --ve=100 --vs=0.001 &

# nohup python main_pl_origin.py --method=dmt --data_name=SwissRoll --batch_size=800 --n_point=800 --perplexity=20 --vs=0.0001 &
# nohup python main_pl_origin.py --method=dmt_mask --data_name=SwissRoll --batch_size=800 --n_point=800 --perplexity=20 --vs=0.0001 &

# python main_pl_aug.py --method=dmt_mask --data_name=Digits --eta=0 --lr=0.0005 --perplexity=20 --pow_input=2 --pow_latent=2 --ve=-1 --vs=0.01 --K=2
# python main_pl_aug.py --method=dmt_mask --data_name=Digits --eta=0 --lr=0.0005 --perplexity=20 --pow_input=2 --pow_latent=2 --ve=-1 --vs=0.01 --K=3
# python main_pl_aug.py --method=dmt_mask --data_name=Digits --eta=0 --lr=0.0005 --perplexity=20 --pow_input=2 --pow_latent=2 --ve=-1 --vs=0.01 --K=4
# python main_pl_aug.py --method=dmt_mask --data_name=Digits --eta=0 --lr=0.0005 --perplexity=20 --pow_input=2 --pow_latent=2 --ve=-1 --vs=0.01 --K=5
# python main_pl_aug.py --method=dmt_mask --data_name=Digits --eta=0 --lr=0.0005 --perplexity=20 --pow_input=2 --pow_latent=2 --ve=-1 --vs=0.01 --K=6
# python main_pl_aug.py --method=dmt_mask --data_name=Digits --eta=0 --lr=0.0005 --perplexity=20 --pow_input=2 --pow_latent=2 --ve=-1 --vs=0.01 --K=7
# python main_pl_aug.py --method=dmt_mask --data_name=Digits --eta=0 --lr=0.0005 --perplexity=20 --pow_input=2 --pow_latent=2 --ve=-1 --vs=0.01 --K=8
# python main_pl_aug.py --method=dmt_mask --data_name=Digits --eta=0 --lr=0.0005 --perplexity=20 --pow_input=2 --pow_latent=2 --ve=-1 --vs=0.01 --K=9
# python main_pl_aug.py --method=dmt_mask --data_name=Digits --eta=0 --lr=0.0005 --perplexity=20 --pow_input=2 --pow_latent=2 --ve=-1 --vs=0.01 --K=10
# python main_pl_aug.py --method=dmt_mask --data_name=Digits --eta=0 --lr=0.0005 --perplexity=20 --pow_input=2 --pow_latent=2 --ve=-1 --vs=0.01 --K=11
# python main_pl_aug.py --method=dmt_mask --data_name=Digits --eta=0 --lr=0.0005 --perplexity=20 --pow_input=2 --pow_latent=2 --ve=-1 --vs=0.01 --K=12
# python main_pl_aug.py --method=dmt_mask --data_name=Digits --eta=0 --lr=0.0005 --perplexity=20 --pow_input=2 --pow_latent=2 --ve=-1 --vs=0.01 --K=13

# python ./autotrain/autotrain_Digits.py
# python ./autotrain/autotrain_MNIST.py
# python ./autotrain/autotrain_Coil20.py
# python ./autotrain/autotrain_KMnist.py
# python ./autotrain/autotrain_SAMUSIK.py
# python ./autotrain/autotrain_HCL6K.py
# python ./autotrain/autotrain_EMNIST.py
# python ./autotrain/autotrain_Coil20.py
# python ./autotrain/autotrain_Coil100.py


CUDA_VISIBLE_DEVICES=0 wandb agent cairi/DLME_ECCV2022/db3n3ovu & 
CUDA_VISIBLE_DEVICES=1 wandb agent cairi/DLME_ECCV2022/db3n3ovu & 
CUDA_VISIBLE_DEVICES=2 wandb agent cairi/DLME_ECCV2022/db3n3ovu & 
CUDA_VISIBLE_DEVICES=3 wandb agent cairi/DLME_ECCV2022/db3n3ovu 
# CUDA_VISIBLE_DEVICES=4 wandb agent cairi/DLME_ECCV2022/db3n3ovu & 
# CUDA_VISIBLE_DEVICES=5 wandb agent cairi/DLME_ECCV2022/db3n3ovu & 
# CUDA_VISIBLE_DEVICES=6 wandb agent cairi/DLME_ECCV2022/db3n3ovu & 
# CUDA_VISIBLE_DEVICES=7 wandb agent cairi/DLME_ECCV2022/db3n3ovu

# python baseline.py --data_name HCL60K
# python baseline.py --data_name HCL280K
# python baseline.py --data_name SAMUSIK
# python baseline.py --data_name PBMC
# python baseline.py --data_name Gast10k
# python baseline.py --data_name Mnist
# python baseline.py --data_name EMnist
# python baseline.py --data_name KMnist
# python baseline.py --data_name Coil20
# python baseline.py --data_name Coil100