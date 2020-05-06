import torch
import torch.nn as nn
from pytorch3d.ops import GraphConv
from gcnna.layers.utils import unpack_mesh_attr, pad_mesh_attr, pack_mesh_attr

class GlobalAveragePooling(nn.Module):
    '''
    Global Average Pooling Layer for Mesh objects
    n * f -> 1 * f 
    '''
    def __init__(self):
        super(GlobalAveragePooling,self).__init__()
        
    def forward(self, verts, verts_idx):
        verts_unpkd, verts_size = pad_mesh_attr(verts_idx, verts)
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
        verts_unpkd, verts_size = pad_mesh_attr(verts_idx, verts)
        out  = torch.max(verts_unpkd, 1, keepdim=True)[0]
        return out.view(verts_size.shape[0], -1)
    
    
class SAGPool(nn.Module):
    '''
    Self Attention Pooling Layer for Mesh objects
    n * f -> n' * f 
    where n' = n*k | k <= 1
    '''
    def __init__(self, in_channels, ratio=0.8, activation = torch.tanh):
        ## TODO: Make it work for batched layer - Hint: modify topk
        super(SAGPool,self).__init__()
        self.in_channels = in_channels
        self.ratio = ratio
        self.attn = GraphConv(in_channels,1)
        self.activation = activation
        
    def update_mesh(self, attn_w, verts, edges, mask_idx):
        '''
        attn_w: attention weights (GConv f -> 1)
        verts: vertices of the mesh
        edges: edges of the mesh
        mask_idx: topk indices of the vertices
        '''
        num_nodes = attn_w.size(0)
        mask = mask_idx.new_full((num_nodes, ), -1)
        i = torch.arange(mask_idx.size(0), dtype=torch.long, device=mask_idx.device)
        mask[mask_idx] = i
        row, col = edges.unbind(1)

        # update the edges
        row, col = mask[row], mask[col]
        mask = (row >= 0) & (col >= 0)
        row, col = row[mask], col[mask]
        edges_updated = torch.stack([row, col], dim=1)
        
        # update the verts
        verts_updated = verts[mask_idx] * self.activation(attn_w[mask_idx]).view(-1, 1)

        return verts_updated, edges_updated

        
    def forward(self, verts, edges, verts_idx, edges_idx):
        
        # Attention forward pass
        attn_w = self.attn(verts, edges)
        V = verts.shape[0] 
            
        # Check for meshes with only one vertex
        if V == 1:
            return verts, edges, verts_idx, edges_idx
                
        # get top V*ratio vertices
        _, mask_idx = torch.topk(attn_w, int(V*self.ratio), dim=0, sorted=False)  
        mask_idx, _ = torch.sort(mask_idx.view(-1))
        # Update the mesh structure after pooling
        verts, edges = self.update_mesh(attn_w, verts, edges, mask_idx)
        
        verts_idx = verts_idx[:verts.shape[0]]
        edges_idx = edges_idx[:edges.shape[0]]

        
        return verts, edges, verts_idx, edges_idx
    
    
#     def forward(self, verts, edges, verts_idx, edges_idx):
        
#         # Attention forward pass
#         attn_w = self.attn(verts, edges)

#         # Unpack packed meshes to list
#         attn_w_unpkd, _  = unpack_mesh_attr(verts_idx, attn_w)
#         verts_unpkd, _  = unpack_mesh_attr(verts_idx, verts)
#         edges_unpkd, _ = unpack_mesh_attr(edges_idx, edges)
#         assert len(attn_w_unpkd) == len(verts_unpkd) == len(edges_unpkd)
        
#         # Process topk pooling on each mesh of the batch
#         B = len(attn_w_unpkd)

#         verts_upd, edges_upd = [], []
#         verts_idx_upd, edges_idx_upd = [], []

#         for i in range(B):
#             V = verts_unpkd[i].shape[0] 
            
#             # Check for meshes with only one vertex
#             if V == 1:
#                 verts_upd.append(verts_unpkd[i])
#                 edges_upd.append(edges_unpkd[i])
#                 verts_idx_upd.append(torch.Tensor([i]*verts_unpkd[i].shape[0]).to(device=verts_idx.device, dtype=verts_idx.dtype))
#                 edges_idx_upd.append(torch.Tensor([i]*edges_unpkd[i].shape[0]).to(device=edges_idx.device, dtype=edges_idx.dtype))
#                 continue
                
#             # get top V*ratio vertices
#             _, mask_idx = torch.topk(attn_w_unpkd[i], int(V*self.ratio), dim=0, sorted=False)  
#             mask_idx, _ = torch.sort(mask_idx.view(-1))
#             # Update the mesh structure after pooling
#             v, e = self.update_mesh(attn_w_unpkd[i], verts_unpkd[i], edges_unpkd[i], mask_idx)
#             verts_upd.append(v)
#             edges_upd.append(e)
#             verts_idx_upd.append(torch.Tensor([i]*v.shape[0]).to(device=verts_idx.device, dtype=verts_idx.dtype))
#             edges_idx_upd.append(torch.Tensor([i]*e.shape[0]).to(device=edges_idx.device, dtype=edges_idx.dtype))
    
#         verts_upd = pack_mesh_attr(verts_upd)
#         edges_upd = pack_mesh_attr(edges_upd)
#         verts_idx_upd = pack_mesh_attr(verts_idx_upd)
#         edges_idx_upd = pack_mesh_attr(edges_idx_upd)
        
#         return verts_upd, edges_upd, verts_idx_upd, edges_idx_upd