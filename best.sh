# # acc 0.96716, con 0.94785
python ./main.py --n_point 60000 --data_name Digits --perplexity 40 --batch_size 1200 --lr 0.001 --vs 0.001 --ve -1 --method dmt --K 10 --num_latent_dim 2 --augNearRate 100000 --name autotrain77/144 --epoch 1500

# # acc 0.8965, con 0.9068
python ./main.py --n_point 60000 --data_name Coil20 --perplexity 40 --batch_size 400 --lr 0.001 --vs 0.001 --ve -1 --method dmt --K 5 --num_latent_dim 2 --augNearRate 100000 --name autotrain92/144 --epoch 1500

# # acc 0.875, con 0.8113
# python ./main.py --data_name Coil100 --perplexity 40 batch_size 400 --lr 0.001 --vs 0.001 --ve -1 --method dmt --K 5 --num_latent_dim 2 --augNearRate 100000 --name autotrain92/144 --epoch 1500

# # acc 0.9534, con 0.8932
# python ./main.py --n_point 60000 --data_name PBMC --perplexity 20 --batch_size 400 --lr 0.0001 --vs 0.001 --ve -1 --method dmt --K 5 --num_latent_dim 2 --augNearRate 100000 --name autotrain4/6 --epoch 1500

# # acc 0.8654, con 0.9325
# python ./main.py --n_point 60000 --data_name Gast10k --perplexity 20 --batch_size 600 --lr 0.001 --vs 0.001 --ve -1 --method dmt --K 5 --num_latent_dim 2 --augNearRate 100000 --name autotrain39/144 --epoch 4000



# CUDA_VISIBLE_DEVICES=1 python main.py --K 100 --num_latent_dim 3 --epochs 15000 --vs 2e-3 --data_name Seversphere --log_interval 150 &
# CUDA_VISIBLE_DEVICES=2 python main.py --K 200 --num_latent_dim 3 --epochs 15000 --vs 2e-3 --data_name Seversphere --log_interval 150 &
# CUDA_VISIBLE_DEVICES=3 python main.py --K 300 --num_latent_dim 3 --epochs 15000 --vs 2e-3 --data_name Seversphere --log_interval 150 &
# CUDA_VISIBLE_DEVICES=4 python main.py --K 400 --num_latent_dim 3 --epochs 15000 --vs 2e-3 --data_name Seversphere --log_interval 150 &
# CUDA_VISIBLE_DEVICES=5 python main.py --K 500 --num_latent_dim 3 --epochs 15000 --vs 2e-3 --data_name Seversphere --log_interval 150 &
# CUDA_VISIBLE_DEVICES=6 python main.py --K 600 --num_latent_dim 3 --epochs 15000 --vs 2e-3 --data_name Seversphere --log_interval 150 &