import torch
import torch.nn as nn
from pytorch3d.ops import GraphConv
from gcnna.layers.utils import unpack_mesh_attr

class GlobalAveragePooling(nn.Module):
    '''
    Global Average Pooling Layer for Mesh objects
    n * f -> 1 * f 
    '''
    def __init__(self):
        super(GlobalAveragePooling,self).__init__()
        
    def forward(self, verts, verts_idx):
        verts_unpkd, verts_size = unpack_mesh_attr(verts_idx, verts)
        out  = torch.sum(verts_unpkd, 1)/verts_size.view(-1,1)
        return out.view(verts_size.shape[0], -1)
    
    
class GlobalMaxPooling(nn.Module):
    '''
    Global Max Pooling Layer for Mesh objects
    n * f -> 1 * f 
    '''
    def __init__(self):
        super(GlobalMaxPooling,self).__init__()
        
    def forward(self, verts, verts_idx):
        verts_unpkd, verts_size = unpack_mesh_attr(verts_idx, verts)
        out  = torch.max(verts_unpkd, 1, keepdim=True)[0]
        return out.view(verts_size.shape[0], -1)
    
    
class SAGPool(nn.Module):
    '''
    Self Attention Pooling Layer for Mesh objects
    n * f -> n' * f 
    where n' = n*k | k <= 1
    '''
    def __init__(self, in_channels, ratio=0.8, activation = torch.tanh):
        super(SAGPool,self).__init__()
        self.in_channels = in_channels
        self.ratio = ratio
        self.attn = GraphConv(in_channels,1)
        self.activation = activation
        
        
    def forward(self, verts, edges, verts_idx, edges_idx):
        # Attention layer
        verts = self.activation(self.attn(verts, edges))
        # Unpack meshes from the packed batch
        verts_unpkd, verts_size  = unpack_mesh_attr(verts_idx, verts)
        edges_unpkd, edges_size = unpack_mesh_attr(edges_idx, edges)
        #
        out  = torch.max(verts_padded, 1, keepdim=True)[0]
        return out.view(verts_size.shape[0], -1)