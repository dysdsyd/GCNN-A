import os, sys
sys.path.append('../')
import argparse
import random 
import numpy as np
from tqdm import tqdm
import torch
import re, yaml, pickle
from torch import nn, optim
from torch.autograd import Variable
from pytorch3d.structures import Meshes
from pytorch3d.utils import ico_sphere
from pytorch3d.ops import sample_points_from_meshes
from pytorch3d.io import load_obj, save_obj
from pytorch3d.loss import mesh_laplacian_smoothing, mesh_edge_loss
from style_transfer.config import Config
from style_transfer.models.base_nn import GraphConvClf
from style_transfer.config import Config
from style_transfer.utils.torch_utils import train_val_split, save_checkpoint, accuracy
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.animation #import FuncAnimation
from matplotlib.animation import FuncAnimation
from ico_objects import ico_disk
mpl.rcParams['savefig.dpi'] = 80
mpl.rcParams['figure.dpi'] = 80
import warnings
warnings.filterwarnings("ignore")
device = torch.device("cuda")
# from IPython.core.display import display, HTML
# display(HTML("<style>.container { width:80% !important; }</style>"))



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
    plt.savefig(title+'.png')
    
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
        ax.view_init(i,i)
    anim = FuncAnimation(fig, update, frames=np.arange(0, 180), interval=100)
    anim.save(title+'.gif', dpi=80, writer = matplotlib.animation.PillowWriter())
    
def gif_mesh_plt(mesh, title=""):
    # Sample points uniformly from the surface of the mesh.
    verts = mesh.verts_packed().cpu()
    faces = mesh.faces_packed().cpu()
    x = verts[:,0]
    y = verts[:,1]
    z = verts[:,2]
    i = faces[:,0]
    j = faces[:,1]
    k = faces[:,2]
    fig = plt.figure(figsize=(5, 5))
    ax = Axes3D(fig)
    ax.plot_trisurf(x, y, z, triangles = [i,j,k])
    ax.set_title(title)
    def update(i):
        ax.view_init(i,i)
    anim = FuncAnimation(fig, update, frames=np.arange(0, 360), interval=100)
    anim.save(title+'_plt.gif', dpi=80, writer = matplotlib.animation.PillowWriter())


def deep_dream_loss(verts, edges, classification_model, layer_name='fc2', feat=None):
        nnfeatures =  classification_model.get_forward_feats(verts, edges, layer_name)
        nnfeatures=torch.squeeze(nnfeatures)
        if 'fc' in layer_name:
            loss = nnfeatures.norm()
        elif 'gconv' in layer_name:
            loss = nnfeatures[:, feat].norm()
        return loss
    
def gconv_loss(verts, edges, faces, classification_model, layer_name='gonv0', feat=None, method='deepdream', target=None):
    ## Check the layer  
    ii = int(layer_name[-1])
    ## if it is a latent layer then run vanilla graph conv first
    if ii != 0: 
        i = ii-1
        verts = classification_model.get_forward_feats(mesh=None,verts=verts, edges=edges ,layername='gconv'+str(i))
            
    ##  Get Center Vertex
    center_verts = verts[0]
    
    ## Get Neighboring Vertices
    n_v = []
    for i in faces:
        if 0 in i:
            n_v += i.tolist()
    n_v = list(set(n_v) - {0})
    neighbor_verts = verts[n_v]
    
    ## Run graph conv for based on one vertex
    center_verts = classification_model.gconvs[ii].w0(center_verts)
    neighbor_verts = classification_model.gconvs[ii].w1(neighbor_verts)
    
    ## Graph Conv
    nnfeatures = center_verts + torch.sum(neighbor_verts, dim=0)
    if method=='feat_inv':
        trg_features =  classification_model.get_forward_feats(mesh=target, layername=layer_name)
        trg_feat = trg_features[trg_features.norm(dim=0).argmax(), :]
        loss = torch.mean((nnfeatures - trg_feat)**2)
    elif method=='deepdream':
        loss = nnfeatures[feat].norm()
    return loss
    

if __name__ == "__main__":
    ## settings
    parser = argparse.ArgumentParser()
    parser.add_argument('-o', '--output_filename', type=str)
    parser.add_argument('-cfg', '--config_path', type=str)
    parser.add_argument('-mpth', '--model_path', type=str)
    parser.add_argument('-alr', '--adam_lr', type=float, default=0.01)
    parser.add_argument('-ab1', '--adam_beta1', type=float, default=0.9)
    parser.add_argument('-bs', '--batch_size', type=int, default=4)
    parser.add_argument('-wm', '--which_starting_mesh', type=str, default='sphere')
    parser.add_argument('-wv', '--which_vertex', type=int, default=None)
    parser.add_argument('-wf', '--which_feature', type=int, default=100)
    parser.add_argument('-wl', '--which_layer', type=str, default='fc1')
    parser.add_argument('-lap', '--mesh_laplacian_smoothing', type=bool, default=True)
    parser.add_argument('-ni', '--num_iteration', type=int, default=400)
    parser.add_argument('-cd', '--camera_distance', type=float, default=2.5)
    parser.add_argument('-ib', '--init_bias', type=str, default='(0,0,0)')
    #parser.add_argument('-g', '--gpu', type=int, default=0)
    args = parser.parse_args()  

    ## Set the device
    device = torch.device("cuda:0")
  
    ## load in trained graph convolutional NN classification model
    graphconv_model_path = args.model_path
    
    idx_best_loss=16 
    LR =1.0
    reduced_LR = 0.1
    reduce_lr_at = 200
    ITERS = args.num_iteration 
    method = 'deepdream'
    target = None

    cfg = Config(args.config_path)
    
    ## SET UP model and optimizer
    classification_model = GraphConvClf(cfg).cuda()
    classification_model.load_state_dict(torch.load(graphconv_model_path+'/model@epoch'+str(idx_best_loss)+'.pkl')['state_dict'])
    
    ## freeze the parameters for the classification model
    for pp in classification_model.parameters():
        pp.requires_grad=False
    
#     which_layer = args.which_layer

    print(args.which_starting_mesh)
    
    ##### Loop over all layers and features
    for which_layer in ['gconv0', 'gconv1', 'gconv2']:
        ii = int(which_layer[-1])
        all_feats=classification_model.gconvs[ii].output_dim
        print(all_feats)
        for which_feature in range(all_feats):
            ## make the output directory if necessary 
            if args.which_starting_mesh=='sphere':
                output_dir = '_'.join(['/scratch/jiadeng_root/jiadeng/dasyed/results_dreaming_sphere/results_dreaming', str(which_layer), str(which_feature)])
            elif args.which_starting_mesh=='plane':
                output_dir = '_'.join(['/scratch/jiadeng_root/jiadeng/dasyed/results_dreaming_plane/results_dreaming', str(which_layer), str(which_feature)])
            elif args.which_starting_mesh=='disk':
                output_dir = '_'.join(['/scratch/jiadeng_root/jiadeng/dasyed/results_dreaming_disk/results_dreaming', str(which_layer), str(which_feature)])
            else:
                print('Please specify a valid input mesh, one of: sphere, plane, or filepath to obj mesh file')
                sys.exit() 

            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
            else:
                print('Warning! skipping over'+str('_'.join([str(which_layer), str(which_feature)])))
                continue

            ## SET UP seed mesh for dreaming
            if args.which_starting_mesh=='sphere':
                ## FOR loading in sphere; initialize the source shape to be a sphere of radius 1
                src_mesh = ico_sphere(4, device=device)
            elif args.which_starting_mesh=='plane':
                ## FOR loading in plane
                src_mesh = ico_plane(2., 3., 2, precision = 1.0, z = 0.0, device=device, color = None)
            elif args.which_starting_mesh=='disk':
                ## FOR loading in disk
                src_mesh = ico_disk(level=2).cuda()
            elif os.path.isfile(args.which_starting_mesh):
                ## FOR loading in input mesh from file
                verts, faces, aux=load_obj(args.which_starting_mesh)
                faces_idx = faces.verts_idx.cuda()
                verts = verts.cuda()
                src_mesh = Meshes(verts=[verts], faces=[faces_idx])
            else:
                print('Please specify a valid input mesh, one of: sphere, plane, or filepath to obj mesh file')
                sys.exit()      
  
            verts = src_mesh.verts_packed()
            faces = src_mesh.faces_packed()
            edges = src_mesh.edges_packed()
            
            plot_period = 20 
            # --------------------------------------------------------------------------------------------
            #   DREAMING LOOP
            # --------------------------------------------------------------------------------------------
            print('\n Deep Dream for %s'%(output_dir))  
            verts = Variable(verts, requires_grad=True)
            classification_model.zero_grad()    
            
            for iter_ in range(ITERS):
                new_verts = verts
#                 loss = deep_dream_loss(new_verts, edges, classification_model, which_layer, which_feature)
                loss = gconv_loss(new_verts, edges, faces, classification_model, which_layer, which_feature, method, target)
                
                #if args.mesh_laplacian_smoothing: 
                    ## add mesh laplacian smoothing
                    #loss_laplacian = mesh_laplacian_smoothing(new_src_mesh, method="uniform")
                    #loss+= 0.1*loss_laplacian
                
                ## Reduce LR
                if iter_ == reduce_lr_at:
                    LR = reduced_LR
                
                ## Plot mesh
                if iter_ % plot_period == 0:
                    if torch.sum(torch.isnan(loss)).item()>0:
                        print('nan values in loss:', torch.sum(torch.isnan(loss)).item())
                    #gif_pointcloud(new_src_mesh, title=os.path.join(output_dir, os.path.splitext(args.output_filename)[0]+"iter_%d" % iter_))
                    print('Iteration: '+str(iter_) + ' Loss: '+str(loss))
                    #
                ## apply loss 
                loss.backward()
                verts.data += LR * verts.grad.data   
            
            new_src_mesh = Meshes([verts.detach()], [faces])
            ## final obj shape
            gif_pointcloud(new_src_mesh, title=os.path.join(output_dir, os.path.splitext(args.output_filename)[0]+"iter_%d" % iter_))  
#             gif_mesh_plt(new_src_mesh, title=os.path.join(output_dir, os.path.splitext(args.output_filename)[0]+"iter_%d" % iter_))  
            ## save output mesh
            save_obj(os.path.join(output_dir,os.path.splitext(args.output_filename)[0]+'.obj'), new_src_mesh.verts_packed(), new_src_mesh.faces_packed())           


