#!/usr/bin/env bash

## run the training
python train.py \
--method edge_cnn \
--dataroot /scratch/jiadeng_root/jiadeng/shared_data/gcnna_data/shrec_16/ \
--name edge_cnn \
--ncf 64 128 256 256 \
--pool_res 600 450 300 180 \
--norm group \
--resblocks 1 \
--lr 0.0001 \
--load_cache false \ 
--flip_edges 0.05 \
--slide_verts 0.2 \
--num_aug 10 \
--niter_decay 100 \