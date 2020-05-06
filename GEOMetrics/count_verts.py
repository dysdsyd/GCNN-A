import torch
from layers import * 
from models import *
from voxel  import voxel2obj
from tqdm import tqdm
from glob import glob
import os, shutil
from torch.utils.data import DataLoader
from pytorch3d.io import load_obj, save_obj
from multiprocessing import Pool
import warnings
warnings.filterwarnings("ignore")
from scripts import binvox_rw
import scipy.io as sio
import numpy as np


path = '/scratch/jiadeng_root/jiadeng/shared_data/datasets/GEOMetrics/shapenet/'
# verts_list = []
for p in tqdm(glob(path+'/*/*')):
#     print(p)
    verts, faces, aux = load_obj(p+'/model.obj')
    if len(verts) > 6000:
        shutil.rmtree(p)
#     verts_list.append(len(verts))
    
# np.save('verts_size.npy', np.array(verts_list))
