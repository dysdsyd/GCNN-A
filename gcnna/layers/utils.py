import torch
import torch.nn as nn
from typing import List, Union
from pytorch3d.structures.utils import packed_to_list, list_to_padded,padded_to_list, list_to_packed

def pad_mesh_attr(idx, attr):
    '''
    packed to padded
    idx: index of each mesh in the batch
    attr:  which attribute to unpack
    '''
    size = idx.unique(return_counts=True)[1]
    lst = packed_to_list(attr, tuple(size))
    padded = list_to_padded(lst)
    return padded, size

def unpack_mesh_attr(idx, attr):
    '''
    packed to padded
    idx: index of each mesh in the batch
    attr:  which attribute to unpack
    '''
    size = idx.unique(return_counts=True)[1]
    lst = packed_to_list(attr, tuple(size))
#     padded = list_to_padded(lst)
    return lst, size

def pack_mesh_attr(attr):
    '''
    list to packed
    idx: index of each mesh in the batch
    attr:  which attribute to pack
    '''
    packed = list_to_packed(attr)[0]
    return packed