# CUDA_VISIBLE_DEVICES=1 wandb agent zangzelin/SCLMAP/l4b0bakq & 
# sleep 10s
# CUDA_VISIBLE_DEVICES=1 wandb agent zangzelin/SCLMAP/l4b0bakq & 
# sleep 10s
# CUDA_VISIBLE_DEVICES=0 wandb agent zangzelin/SCLMAP/l4b0bakq & 
# sleep 10s
# CUDA_VISIBLE_DEVICES=3 wandb agent zangzelin/SCLMAP/l4b0bakq &
# sleep 10s
# CUDA_VISIBLE_DEVICES=4 wandb agent zangzelin/SCLMAP/l4b0bakq & 
# sleep 10s
# CUDA_VISIBLE_DEVICES=5 wandb agent zangzelin/SCLMAP/l4b0bakq & 
# sleep 10s
# CUDA_VISIBLE_DEVICES=6 wandb agent zangzelin/SCLMAP/l4b0bakq & 
# sleep 10s
# CUDA_VISIBLE_DEVICES=7 wandb agent zangzelin/SCLMAP/l4b0bakq &

# sleep 300s

# CUDA_VISIBLE_DEVICES=0 wandb agent zangzelin/SCLMAP/l4b0bakq & 
# sleep 10s
# CUDA_VISIBLE_DEVICES=1 wandb agent zangzelin/SCLMAP/l4b0bakq & 
# sleep 10s
# CUDA_VISIBLE_DEVICES=2 wandb agent zangzelin/SCLMAP/l4b0bakq & 
# sleep 10s
# CUDA_VISIBLE_DEVICES=3 wandb agent zangzelin/SCLMAP/l4b0bakq &
# sleep 10s
# CUDA_VISIBLE_DEVICES=4 wandb agent zangzelin/SCLMAP/l4b0bakq & 
# sleep 10s
# CUDA_VISIBLE_DEVICES=5 wandb agent zangzelin/SCLMAP/l4b0bakq & 
# sleep 10s
# CUDA_VISIBLE_DEVICES=6 wandb agent zangzelin/SCLMAP/l4b0bakq & 
# sleep 10s
# CUDA_VISIBLE_DEVICES=7 wandb agent zangzelin/SCLMAP/l4b0bakq &


# CUDA_VISIBLE_DEVICES=0 python sclmap_main_aug_hop.py --v_latent 0.001 
# CUDA_VISIBLE_DEVICES=1 python sclmap_main_aug_hop.py --v_latent 0.01
CUDA_VISIBLE_DEVICES=1 python sclmap_main_aug_hop.py --v_latent 0.1
CUDA_VISIBLE_DEVICES=1 python sclmap_main_aug_hop.py --v_latent 1
CUDA_VISIBLE_DEVICES=1 python sclmap_main_aug_hop.py --v_latent 10
CUDA_VISIBLE_DEVICES=1 python sclmap_main_aug_hop.py --v_latent 100