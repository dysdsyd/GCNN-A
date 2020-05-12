import torch
from torch.autograd import Variable
from torch.optim import SGD
from tqdm import tqdm
import os, sys
from pytorch3d.utils import ico_sphere
try:
    from .ico_objects import ico_disk
except:
    from ico_objects import ico_disk
from pytorch3d.io import load_obj, save_obj
from pytorch3d.ops import sample_points_from_meshes
from pytorch3d.structures import Meshes
from pytorch3d.loss import (
    chamfer_distance, 
    mesh_edge_loss, 
    mesh_laplacian_smoothing, 
    mesh_normal_consistency,
)

from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.animation #import FuncAnimation
from matplotlib.animation import FuncAnimation
mpl.rcParams['savefig.dpi'] = 80
mpl.rcParams['figure.dpi'] = 80

device = torch.device("cuda")

import warnings
warnings.filterwarnings("ignore")

cos = torch.nn.CosineSimilarity(dim=0)

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

def gif_pointcloud(mesh, path=""):
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
    anim.save(os.path.join(path, 'model.gif'), dpi=80, writer = matplotlib.animation.PillowWriter())

class FeatureVisualization():
    def __init__(self, model, src_mesh, exp):
        """
        model: Trained Model
        src_mesh: sphere or disk
        exp: name of hte experiment
        """
        self.model = model
        self.model.eval()
        self.src_mesh_name = src_mesh
        self.exp = exp
        # self.result_dir = os.path.join('/scratch/jiadeng_root/jiadeng/shared_data/gcnna_data/', self.exp)
        self.result_dir = os.path.join('results/gcnna_data/', self.exp)
        os.makedirs(self.result_dir, exist_ok=True)
        
    def load_mesh(self, pth):
        verts, faces, aux = load_obj(pth)
        mesh = Meshes(verts=[verts], faces=[faces.verts_idx])
        return mesh.cuda()

    def normalize_verts(self, verts):
        # X
        if (verts[:,0].max() - verts[:,0].min()) != 0:
            verts[:,0] = ((verts[:,0] - verts[:,0].min())/(verts[:,0].max() - verts[:,0].min())) - 0.5
        else:
            verts[:,0] = 0.1

        # Y
        if (verts[:,1].max() - verts[:,1].min()) != 0:
            verts[:,1] = ((verts[:,1] - verts[:,1].min())/(verts[:,1].max() - verts[:,1].min())) - 0.5
        else:
            verts[:,1] = 0.1

        # Z
        if (verts[:,2].max() - verts[:,2].min()) != 0:
            verts[:,2] = ((verts[:,2] - verts[:,2].min())/(verts[:,2].max() - verts[:,2].min())) - 0.5
        else:
            verts[:,2] = 0.1
        return verts

    def PCA_svd(self, X, k=3, center=True):
        n = X.size()[0]
        ones = torch.ones(n).view([n,1])
        h = ((1/n) * torch.mm(ones, ones.t())) if center  else torch.zeros(n*n).view([n,n])
        H = (torch.eye(n) - h).cuda()
        X_center =  torch.mm(H.double(), X.double())
        u, s, v = torch.svd(X_center) 
    #     print(u.t().shape, v.t().shape)
        components  = v[:k].t()
        explained_variance = torch.mul(s[:k], s[:k])/(n-1)
        return components
                

    def alpha_norm(self, input_matrix, alpha):
        """
            Converts matrix to vector then calculates the alpha norm
        """
        alpha_norm = ((input_matrix.view(-1))**alpha).sum()
        return alpha_norm


    def euclidian_loss(self, src_feats, trg_feats):
        """
            Euclidian loss is the main loss function in the paper
            ||fi(x) - fi(x_0)||_2^2& / ||fi(x_0)||_2^2
        """
        distance_matrix = trg_feats - src_feats
        euclidian_distance = self.alpha_norm(distance_matrix, 2)
        normalized_euclidian_distance = euclidian_distance / self.alpha_norm(src_feats, 2)
        return normalized_euclidian_distance
    
    def gram_matrix(self, feats):
        '''
        feats: N * F
        returns: F * F matrix 
        '''
        N, _ = feats.shape
        return torch.mm(feats.T, feats)/N
    
    def get_gram_loss(self, src_feats, trg_feats):
        src_gram = self.gram_matrix(src_feats)
        trg_gram = self.gram_matrix(trg_feats)
#         print(src_gram.shape, trg_gram.shape)
        return torch.sum((src_gram - trg_gram)**2)
    
    def rmse(self, src_feats, trg_feats):
        return torch.sqrt(torch.mean((src_feats - trg_feats)**2))

    def get_feats_from_layer(self, x, layer_id):
        """
            Saves the output after a forward pass until nth layer
            This operation could be done with a forward hook too
            but this one is simpler (I think)
        """
        layer_output = self.model.extract_feats(x, layer_id)
        return layer_output

    ################### Feature Inversion ######################
    def invert_feats(self, trg_obj_path, trg_feats=None, layer='latent',filter=None, lr=1, weights=None, iters = 200, ico_level=3, verbose=False):
        # Load intial mesh
        if self.src_mesh_name == "sphere":
            self.src_mesh = ico_sphere(ico_level).cuda()
        elif self.src_mesh_name =='disk':
            self.src_mesh = ico_disk(ico_level).cuda()
        else:
            self.src_mesh = self.load_mesh(src_mesh)   

        # Load target mesh
        if trg_feats == None:
            self.trg_mesh = self.load_mesh(trg_obj_path)
            # Get the output from the model after a forward pass until target_layer with the target object
            trg_feats = self.get_feats_from_layer(self.trg_mesh, layer)
            self.trg_feats = trg_feats
        
        # optimize defrom_verts
        deform_verts = Variable(self.normalize_verts(self.src_mesh.verts_packed().clone()), requires_grad = True)
        optimizer = torch.optim.Adam([deform_verts], lr = lr)

        for i in tqdm(range(iters)):
            optimizer.zero_grad()
            self.new_src_mesh = Meshes(verts=[deform_verts], faces=[self.src_mesh.faces_packed()])
            # Get the output from the model after a forward pass until target_layer for the source object
            src_feats = self.get_feats_from_layer(self.new_src_mesh, layer)


            # Losses
            # cosine_loss = cos(trg_feats.detach(), src_feats)
            # euc_loss = .1*self.euclidian_loss(trg_feats.detach(), src_feats)
            # rmse_loss = self.rmse(trg_feats.detach(), src_feats)
            # latent_loss = 0.000005*(torch.mean(torch.abs(trg_feats.detach() - src_feats)))
            # gram_loss = 1e-17*self.get_gram_loss(src_feats, trg_feats.detach())
            if filter == None:
                cd_loss, _ = chamfer_distance(trg_feats.unsqueeze(0).detach(), src_feats.unsqueeze(0))
                # cd_loss, _ = chamfer_distance(self.PCA_svd(trg_feats, k=10).t().unsqueeze(0).detach(), 
                #             self.PCA_svd(src_feats,k=10).t().unsqueeze(0))
            else:
                # print(trg_feats[:,filter].view(1,-1,1).shape)
                cd_loss, _ = chamfer_distance( trg_feats[:,filter].view(1,-1,1).detach(), src_feats[:,filter].view(1,-1,1))
            cd_loss = weights['cd_loss'] * cd_loss
            
            # Regularizations
            laplacian_loss = weights['lap_loss']*mesh_laplacian_smoothing(self.new_src_mesh, method="uniform")
            edge_loss = weights['edge_loss']*mesh_edge_loss(self.new_src_mesh)
            # normal_loss = 1*mesh_normal_consistency(self.new_src_mesh)
            
            # Sum all to optimize
            loss =  cd_loss + laplacian_loss + edge_loss#+ gram_loss #+ gram_loss  #normal_loss #+latent_loss + cosine_loss + euc_loss#+#euc_loss  edge_loss
            
            # Step
            loss.backward()
            optimizer.step()
            
            # Generate image every 5 iterations
            if i % int(iters/5) == 0 and verbose:
                print('Iteration:', str(i), 
                      'Loss:', loss.item(), 
                      'CD Loss', cd_loss.item(),
                    #   "Gram Loss:", gram_loss.item(),
#                       "Cosine Loss:", cosine_loss.item(),
#                       "Latent Loss:", latent_loss.item(),
#                       "Euc Loss:", euc_loss.item(), 
                      "Lap Loss:", laplacian_loss.item(), 
                      "Edge Loss:", edge_loss.item(),
                    #  ,"Normal Loss:", normal_loss.item(),
                     )
                
                plot_pointcloud(self.new_src_mesh)
                

            # Reduce learning rate every 50 iterations
            # if i % 100 == 0:
            #     for param_group in optimizer.param_groups:
            #         param_group['lr'] *= 1/10
                    
        os.makedirs(self.result_dir, exist_ok=True)
        if filter == None:
            obj_path = os.path.join(self.result_dir, self.src_mesh_name+'_'+layer+'.obj')
        else:
            obj_path = os.path.join(self.result_dir, self.src_mesh_name + '_' + layer + '_' + str(filter) + '.obj') 
        save_obj(obj_path, self.new_src_mesh.verts_packed(), self.new_src_mesh.faces_packed())

    
    ################### DeepDream per layer for each channel ######################
    def dream_layer(self, layer, filter, iters=200, ico_level=3):
        # Load intial mesh
        if self.src_mesh_name == "sphere":
            self.src_mesh = ico_sphere(ico_level).cuda()
        elif self.src_mesh_name =='disk':
            self.src_mesh = ico_disk(ico_level).cuda()
        else:
            self.src_mesh = self.load_mesh(src_mesh)    
        
        deform_verts = Variable(self.normalize_verts(self.src_mesh.clone().verts_packed()), requires_grad = True)
        optimizer = torch.optim.Adam([deform_verts], lr = 0.01)
        
        for i in tqdm(range(iters)):
            optimizer.zero_grad()
            self.new_src_mesh = Meshes(verts=[deform_verts], faces=[self.src_mesh.faces_packed()])
            # Get the output from the model after a forward pass until target_layer for the source object
            src_feats = self.get_feats_from_layer(self.new_src_mesh, layer)[:, filter]
            
            # Regularizations
            laplacian_loss = 0.1*mesh_laplacian_smoothing(self.new_src_mesh, method="uniform")
            edge_loss = 0.1**mesh_edge_loss(self.new_src_mesh)
            # normal_loss = 1*mesh_normal_consistency(self.new_src_mesh)
            verts_reg = 1*(torch.mean(torch.abs(deform_verts[:,0])) + 
                            torch.mean(torch.abs(deform_verts[:,1])) + 
                            torch.mean(torch.abs(deform_verts[:,2])))
            
            # Losses
            dream_loss = -torch.mean(src_feats)
            # Sum all to optimize
            loss =   dream_loss + verts_reg #+ laplacian_loss + edge_loss 
            # Step
            loss.backward()
            optimizer.step()
            
            # Generate image every 5 iterations
            if i % int(iters/5) == 0:
                print('Iteration:', str(i), 
                      'Loss:', loss.item(), 
                      "Dream Loss", dream_loss.item(),
                      "Lap Loss:", laplacian_loss.item(), 
                      "Edge Loss:", edge_loss.item(),
                      "Verts Reg:", verts_reg.item(),
                     # ,"Normal Loss:", normal_loss.item())
                     )
                plot_pointcloud(self.new_src_mesh)
                plt.close()

            # # Reduce learning rate every 50 iterations
            # if i % 100 == 0:
            #     for param_group in optimizer.param_groups:
            #         param_group['lr'] *= 1/5








