#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.


import torch
from pytorch3d.ops.subdivide_meshes import SubdivideMeshes
from pytorch3d.structures.meshes import Meshes
import pdb 
import math

# Vertex coordinates for a level 0 ico-disk.
_ico_verts0 = [
    [0.1000, 0.1000, 1.0000],
    [1.1000, 0.1000, 1.0000],
    [0.8071, 0.8071, 1.0000],
    [0.1000, 1.1000, 1.0000],
    [-0.6071, 0.8071, 1.0000],
    [-0.9000, 0.1000, 1.0000],
    [-0.6071, -0.6071, 1.0000],    
    [0.1000, -0.9000, 1.0000],
    [0.8071, -0.6071, 1.0000],
]


# Faces for level 0 ico-disk
_ico_faces0 = [
    [0,1,2],
    [0,2,3],
    [0,3,4],
    [0,4,5],
    [0,5,6],
    [0,6,7],
    [0,7,8],
    [0,8,1],
]

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

def ico_disk(level: int = 0, device=None):
    """
    Create verts and faces for a unit ico-disk, with all faces oriented
    consistently.
    Args:
        level: integer specifying the number of iterations for subdivision
               of the mesh faces. Each additional level will result in four new
               faces per face.
        device: A torch.device object on which the outputs will be allocated.
    Returns:
        Meshes object with verts and faces.
    """
    if device is None:
        device = torch.device("cpu")
    if level < 0:
        raise ValueError("level must be >= 0.")
    if level == 0:
        verts = torch.tensor(_ico_verts0, dtype=torch.float32, device=device)
        faces = torch.tensor(_ico_faces0, dtype=torch.int64, device=device)

    else:
        mesh = ico_disk(level - 1, device)
        subdivide = SubdivideMeshes()
        mesh = subdivide(mesh)
        verts = mesh.verts_list()[0]
        faces = mesh.faces_list()[0]
    return Meshes(verts=[verts], faces=[faces])
    
    # Append the center vertex

#def ico_plane(level: int = 0, device=None):
#    """
#    Create verts and faces for a unit ico-sphere, with all faces oriented
#    consistently.
#    Args:
#        level: integer specifying the number of iterations for subdivision
#               of the mesh faces. Each additional level will result in four new
#               faces per face.
#        device: A torch.device object on which the outputs will be allocated.
#    Returns:
#        Meshes object with verts and faces.
#    """
#    if device is None:
#        device = torch.device("cpu")
#    if level < 0:
#        raise ValueError("level must be >= 0.")
#    if level == 0:
#        verts = torch.tensor(_ico_verts0, dtype=torch.float32, device=device)
#        faces = torch.tensor(_ico_faces0, dtype=torch.int64, device=device)
#
#    else:
#        mesh = ico_sphere(level - 1, device)
#        subdivide = SubdivideMeshes()
#        mesh = subdivide(mesh)
#        verts = mesh.verts_list()[0]
#        verts /= verts.norm(p=2, dim=1, keepdim=True)
#        faces = mesh.faces_list()[0]
#    return Meshes(verts=[verts], faces=[faces])