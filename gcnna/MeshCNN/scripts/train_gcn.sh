#!/usr/bin/env bash

## run the training
python train.py \
--method gcn_cnn \
--dataroot /scratch/jiadeng_root/jiadeng/shared_data/gcnna_data/shrec_16/ \
--name gcn_cnn \
--ncf 32 64 128 128 \
--ninput_edges 252 \
--pool_res 600 450 300 180 \
--norm group \
--resblocks 1 \
--load_cache false \ 
--flip_edges 0.2 \
--slide_verts 0.2 \
--num_aug 20 \
--niter_decay 100 \