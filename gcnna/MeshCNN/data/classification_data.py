import os, sys
import torch
from .base_dataset import BaseDataset
sys.path.append('../util')
sys.path.append('../models')
try:
    from util.util import is_mesh_file, pad, normalize_verts
except:
    from ..util.util import is_mesh_file, pad, normalize_verts
try:
    from models.layers.mesh import Mesh
except:
    from ..models.layers.mesh import Mesh

class ClassificationData(BaseDataset):

    def __init__(self, opt):
        BaseDataset.__init__(self, opt)
        self.opt = opt
        self.device = torch.device('cuda:{}'.format(opt.gpu_ids[0])) if opt.gpu_ids else torch.device('cpu')
        self.root = opt.dataroot
        self.dir = os.path.join(opt.dataroot)
        self.classes, self.class_to_idx = self.find_classes(self.dir)
        self.paths = self.make_dataset_by_class(self.dir, self.class_to_idx, opt.phase)
        self.nclasses = len(self.classes)
        self.size = len(self.paths)
        self.get_mean_std()
        # modify for network later.
        opt.nclasses = self.nclasses
        if self.opt.method == 'edge_cnn':
            opt.input_nc = self.ninput_channels
        else:
            opt.input_nc = 3
            self.ninput_channels = 3

    def __getitem__(self, index):
        path = self.paths[index][0]
        label = self.paths[index][1]
        mesh = Mesh(file=path, opt=self.opt, hold_history=False, export_folder=self.opt.export_folder)
        meta = {'mesh': mesh, 'label': label}
        # get edge features
        features = mesh.extract_features()
        if self.opt.method == 'edge_cnn':
            features = pad(features, self.opt.ninput_edges)
            meta['features'] = ((features - self.mean) / self.std).unsqueeze(0).float()
        else:
            features = pad(features, 252, dim=0, method='gcn_cnn') # Hardcoded right now based on the data
            # meta['features'] = normalize_verts(features).unsqueeze(0)
            meta['features'] = features.unsqueeze(0)
            meta['edges'] = mesh.edges.unsqueeze(0)
            if self.opt.method =='zgcn_cnn':
                meta['adj'] = mesh.adj.unsqueeze(0)
        return meta

    def __len__(self):
        return self.size

    # this is when the folders are organized by class...
    @staticmethod
    def find_classes(dir):
        classes = [d for d in os.listdir(dir) if os.path.isdir(os.path.join(dir, d))]
        classes.sort()
        class_to_idx = {classes[i]: i for i in range(len(classes))}
        return classes, class_to_idx

    @staticmethod
    def make_dataset_by_class(dir, class_to_idx, phase):
        meshes = []
        dir = os.path.expanduser(dir)
        for target in sorted(os.listdir(dir)):
            d = os.path.join(dir, target)
            if not os.path.isdir(d):
                continue
            for root, _, fnames in sorted(os.walk(d)):
                for fname in sorted(fnames):
                    if is_mesh_file(fname) and (root.count(phase)==1):
                        path = os.path.join(root, fname)
                        item = (path, class_to_idx[target])
                        meshes.append(item)
        return meshes
