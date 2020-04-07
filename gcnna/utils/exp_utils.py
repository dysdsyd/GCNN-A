import pickle
import pandas as pd
import numpy as np
import torch
import matplotlib.pyplot as plt
import yaml
from PyGEL3D import gel
from PyGEL3D import js
import re
from pytorch3d.io import load_obj, save_obj
from pytorch3d.utils import ico_sphere
from pytorch3d.structures import Meshes
from pytorch3d.loss import (
    chamfer_distance, 
    mesh_edge_loss, 
    mesh_laplacian_smoothing, 
    mesh_normal_consistency,
)
from pytorch3d.ops import sample_points_from_meshes
from pytorch3d.structures.utils import packed_to_list, list_to_packed, list_to_padded
from torch.utils.data import DataLoader
from style_transfer.config import Config
from style_transfer.models.base_nn import GraphConvClf
from style_transfer.utils.metrics import compare_meshes, laplacian_loss
# from style_transfer.data.datasets import ShapenetDataset
from style_transfer.utils.torch_utils import EarlyStopping
from tqdm import tqdm_notebook
import warnings
device = torch.device("cuda")
warnings.filterwarnings("ignore")
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.animation #import FuncAnimation
from matplotlib.animation import FuncAnimation
mpl.rcParams['savefig.dpi'] = 80
mpl.rcParams['figure.dpi'] = 80
    
from pytorch3d.ops.subdivide_meshes import SubdivideMeshes
import pdb 
import math





def plot_pointcloud(mesh, title=""):
    # Sample points uniformly from the surface of the mesh.
    points = sample_points_from_meshes(mesh, 5000)
    x, y, z = points.clone().detach().cpu().squeeze().unbind(1)    
    fig = plt.figure(figsize=(5, 5))
    ax = Axes3D(fig)
    ax.scatter3D(x, z, -y)
    ax.set_xlabel('x')
    ax.set_ylabel('z')
    ax.set_zlabel('y')
    ax.set_title(title)
    ax.view_init(190, 30)
    plt.show()
    
def gif_pointcloud(mesh, title=""):
    # Sample points uniformly from the surface of the mesh.
    points = sample_points_from_meshes(mesh, 5000)
    x, y, z = points.clone().detach().cpu().squeeze().unbind(1)    
    fig = plt.figure(figsize=(5, 5))
    ax = Axes3D(fig)
    ax.scatter3D(x, z, -y)
    ax.set_xlabel('x')
    ax.set_ylabel('z')
    ax.set_zlabel('y')
    ax.set_title(title)
    def update(i):
        ax.view_init(190,i)
    anim = FuncAnimation(fig, update, frames=np.arange(0, 100), interval=200)
    return anim
    
    
def load_mesh(tst_obj):
    verts, faces, aux = load_obj(tst_obj)
    mesh = Meshes(verts=[verts], faces=[faces.verts_idx]).cuda()
    plot_pointcloud(mesh)
    return mesh

def gram_matrix(mesh_feat, normalize=True):
    F = mesh_feat.verts_packed()
    V, C = F.size()
    G = torch.mm(F.t(), F)
    if normalize:
        return G/V
    else:
        return G

def gram_loss(feats, target_feats, style_weights):
    gram_loss = 0
    for f, f_target, weight in zip(feats, target_feats, style_weights):
        gram_f = gram_matrix(f)
        gram_t = gram_matrix(f_target)
        gram_loss += weight * torch.sum((gram_f - gram_t)**2)
    return gram_loss

def l2_loss(feat, feat_target):
    return torch.sum((feat - feat_target)**2)

def plot_mesh(mesh):
    save_obj('mesh.obj', mesh.verts_packed(), mesh.faces_packed())
    js.set_export_mode()
    m = gel.obj_load('mesh.obj')
    js.display(m, smooth=False)
    
    
def ico_plane(width, height, num_verts, precision = 1.0, z = 0.0, color = None, device=None):
    offset = 0#num_verts
    normal = [0., 0., 1.]

    w = math.ceil(width / precision)
    h = math.ceil(height / precision)

    ## get vertices
    vertices_ = []
    for y in range(0, int(h)):
        for x in range(0, int(w)):
            offsetX = 0
            if x%2 == 1:
                offsetY = 0.5 
            else:
                offsetY = 0.0
            vertices_.append( [(x + offsetX) * precision, (y + offsetY) * precision, z] )
            #if color:
            #    mesh.addColor( color )
            #mesh.addNormal( normal )
            #mesh.addTexCoord( [float(x+offsetX)/float(w-1), float(y+offsetY)/float(h-1)] )
    faces_=[]
    for y in range(0, int(h)-1):
        for x in range(0, int(w)-1):
            if x%2 == 0:
                faces_.append([offset + x + y * w, offset + (x + 1) + y * w, offset + x + (y + 1) * w])         # d
                faces_.append([offset + (x + 1) + y * w, offset + (x + 1) + (y + 1) * w ,offset + x + (y + 1) * w ])         # d
            else:
                faces_.append([offset + (x + 1) + (y + 1) * w, offset + x + y * w, offset + (x + 1 ) + y * w])        # b
                faces_.append([offset + (x + 1) + (y + 1) * w, offset + x + (y + 1) * w, offset + x + y * w])               # a
    verts = torch.tensor(vertices_, dtype=torch.float32, device=device)
    faces = torch.tensor(faces_, dtype=torch.int64, device=device)
    #pdb.set_trace()
    return Meshes(verts=[verts], faces=[faces])