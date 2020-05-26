#!/usr/bin/env bash

## run the training
python train.py \
--method zgcn_cnn \
--dataroot /scratch/jiadeng_root/jiadeng/shared_data/gcnna_data/shrec_16/ \
--name zgcn_cnn \
--ncf 64 128 256 512 512  \
--ninput_edges 252 \
--pool_res 600 450 300 180 \
--norm group \
--lr 0.0002 \
--resblocks 1 \
--load_cache false \ 
--flip_edges 0.05 \
--slide_verts 0.2 \
--num_aug 10 \
--niter_decay 100 \