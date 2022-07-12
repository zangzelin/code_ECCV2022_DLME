# CUDA_VISIBLE_DEVICES=1 python sclmap_main_aug_hop.py &
# sleep 10s
# # CUDA_VISIBLE_DEVICES=1 wandb cairi/SCLMAP/wd09c8pe 
# # sleep 10s

# CUDA_VISIBLE_DEVICES=0 python sclmap_main_aug_hop.py 
# sleep 10s


CUDA_VISIBLE_DEVICES=0 python sclmap_main_aug_hop.py --seed 1 --v_latent=1 &
CUDA_VISIBLE_DEVICES=0 python sclmap_main_aug_hop.py --seed 2 --v_latent=1 &
CUDA_VISIBLE_DEVICES=0 python sclmap_main_aug_hop.py --seed 3 --v_latent=1 &
CUDA_VISIBLE_DEVICES=0 python sclmap_main_aug_hop.py --seed 4 --v_latent=1 &
CUDA_VISIBLE_DEVICES=0 python sclmap_main_aug_hop.py --seed 5 --v_latent=1 &
CUDA_VISIBLE_DEVICES=0 python sclmap_main_aug_hop.py --seed 6 --v_latent=1 &