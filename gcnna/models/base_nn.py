import torch
import torch.nn as nn
from torch.nn import functional as F
from pytorch3d.ops import GraphConv
from gcnna.config import Config
from gcnna.layers.pooling import GlobalAveragePooling, GlobalMaxPooling, SAGPool
from gcnna.layers.norm import BatchNorm
from pytorch3d.structures.utils import packed_to_list, list_to_padded

class GraphConvClf(nn.Module):
    def __init__(self, cfg):
        super(GraphConvClf, self).__init__()
        input_dim = cfg.GCC.INPUT_MESH_FEATS
        hidden_dims = cfg.GCC.HIDDEN_DIMS 
        classes = cfg.GCC.CLASSES
        gconv_init = cfg.GCC.CONV_INIT
        ratio = cfg.GCC.RATIO
        BN = cfg.GCC.BN
        
        # Graph Convolution Network
        self.gconvs = nn.ModuleList()
        dims = [input_dim] + hidden_dims
        for i in range(len(dims)-1):
            self.gconvs.append(SAGPool(dims[i], ratio=ratio))
            self.gconvs.append(GraphConv(dims[i], dims[i+1], init=gconv_init, directed=False))
            if BN:
                self.gconvs.append(BatchNorm(dims[i+1]))
            
        
        self.gap = GlobalAveragePooling()
        self.gmp = GlobalMaxPooling()
        self.fc1 = nn.Linear(dims[-1]*2,dims[-1]*2)
        self.fc2 = nn.Linear(dims[-1]*2, classes)
        
        nn.init.normal_(self.fc1.weight, mean=0.0, std=0.01)
        nn.init.constant_(self.fc1.bias, 0)
        
        nn.init.normal_(self.fc2.weight, mean=0.0, std=0.01)
        nn.init.constant_(self.fc2.bias, 0)
        
    def forward(self, mesh):
        verts = mesh.verts_packed()
        edges = mesh.edges_packed()
        verts_idx = mesh.verts_packed_to_mesh_idx()
        edges_idx = mesh.edges_packed_to_mesh_idx()
        
        for gconv in self.gconvs:
            if 'SAGPool' in str(gconv):
                verts, edges, verts_idx, edges_idx = gconv(verts, edges, verts_idx, edges_idx)
            elif 'GraphConv'in str(gconv):
                verts = gconv(verts, edges)
            elif 'BatchNorm'in str(gconv):
                verts = gconv(verts)
            verts = F.relu(verts)
            
        
        avg_pool = self.gap(verts, verts_idx)
        max_pool = self.gmp(verts, verts_idx)
        out = torch.cat([avg_pool, max_pool], dim=1)
        out = F.relu(self.fc1(out))
        out =  self.fc2(out)
        return out
    
    def get_forward_feats(self, mesh=None, verts=None, edges=None, layername='gconv0'):
        #graph_features = {}
        if mesh != None:
            verts = mesh.verts_packed()
            edges = mesh.edges_packed()
        
        for ii, gconv in enumerate(self.gconvs):
            verts = F.relu(gconv(verts, edges))
#             pre_verts = gconv(verts, edges)
            if layername=='gconv'+str(ii):
                return verts
#             verts = F.relu(pre_verts)
        
#         ### VERTS ###
#         verts_idx = mesh.verts_packed_to_mesh_idx()
#         verts_size = verts_idx.unique(return_counts=True)[1]
#         verts_packed = packed_to_list(verts, tuple(verts_size))
#         verts_padded = list_to_padded(verts_packed)
        
#         out_mean  = torch.mean(verts, 1)#torch.sum(verts_padded, 1)/verts_size.view(-1,1)
#         out_fc1 = F.relu(self.fc1(out_mean))
#         out_fc2 =  self.fc2(out_fc1)

#         if layername=='mean':       
#             return out_mean
#         elif layername=='fc1':
#             return out_fc1
#         elif layername=='fc2':
#             return out_fc2