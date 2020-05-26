import numpy as np
import os
import ntpath
import torch
from pytorch3d.io import load_obj, save_obj


def fill_mesh(mesh2fill, file: str, opt):
    load_path = get_mesh_path(file, opt.num_aug)
    if os.path.exists(load_path) and opt.load_cache:
        mesh_data = np.load(load_path, encoding='latin1', allow_pickle=True) 
    else:
        mesh_data = from_scratch(file, opt)
        np.savez_compressed(load_path, gemm_edges=mesh_data.gemm_edges, vs=mesh_data.vs, edges=mesh_data.edges,
                            edges_count=mesh_data.edges_count, ve=mesh_data.ve, v_mask=mesh_data.v_mask,
                            filename=mesh_data.filename, sides=mesh_data.sides,
                            edge_lengths=mesh_data.edge_lengths, edge_areas=mesh_data.edge_areas,
                            features=mesh_data.features)
    mesh2fill.vs = mesh_data['vs']
    mesh2fill.edges = mesh_data['edges']
    mesh2fill.gemm_edges = mesh_data['gemm_edges']
    mesh2fill.edges_count = int(mesh_data['edges_count'])
    mesh2fill.ve = mesh_data['ve']
    mesh2fill.v_mask = mesh_data['v_mask']
    mesh2fill.filename = str(mesh_data['filename'])
    mesh2fill.edge_lengths = mesh_data['edge_lengths']
    mesh2fill.edge_areas = mesh_data['edge_areas']
    mesh2fill.features = mesh_data['features']
    mesh2fill.sides = mesh_data['sides']
    mesh2fill.faces = mesh_data['faces']
    mesh2fill.adj = mesh_data['adj']

def get_mesh_path(file: str, num_aug: int):
    filename, _ = os.path.splitext(file)
    dir_name = os.path.dirname(filename)
    prefix = os.path.basename(filename)
    load_dir = os.path.join(dir_name, 'cache')
    load_file = os.path.join(load_dir, '%s_%03d.npz' % (prefix, np.random.randint(0, num_aug)))
    if not os.path.isdir(load_dir):
        os.makedirs(load_dir, exist_ok=True)
    return load_file

def from_scratch(file, opt):

    class MeshPrep:
        def __getitem__(self, item):
            return eval('self.' + item)

    mesh_data = MeshPrep()
    mesh_data.vs = mesh_data.edges = mesh_data.adj = None
    mesh_data.gemm_edges = mesh_data.sides = None
    mesh_data.edges_count = None
    mesh_data.ve = None
    mesh_data.v_mask = None
    mesh_data.filename = 'unknown'
    mesh_data.edge_lengths = None
    mesh_data.edge_areas = []
    mesh_data.vs, mesh_data.faces = fill_from_file(mesh_data, file)
    mesh_data.v_mask = torch.ones(len(mesh_data.vs), dtype=bool)
    mesh_data.faces, face_areas = remove_non_manifolds(mesh_data, mesh_data.faces)
    if opt.num_aug > 1:
        mesh_data.faces = augmentation(mesh_data, opt, mesh_data.faces)
    build_gemm(mesh_data, mesh_data.faces.numpy(), face_areas.numpy())
    if opt.num_aug > 1:
        post_augmentation(mesh_data, opt)
    if opt.method == 'edge_cnn':
        mesh_data.features = extract_features(mesh_data)
    else:
        mesh_data.features = mesh_data.vs
        mesh_data.edges = mesh_data.edges.long()
    
    if opt.method == 'zgcn_cnn':
        mesh_data.adj = calc_adj(mesh_data.faces).float()

    return mesh_data

def fill_from_file(mesh, file):
    mesh.filename = ntpath.split(file)[1]
    mesh.fullfilename = file
    vs, faces = [], []
    f = open(file)
    for line in f:
        line = line.strip()
        splitted_line = line.split()
        if not splitted_line:
            continue
        elif splitted_line[0] == 'v':
            vs.append([float(v) for v in splitted_line[1:4]])
        elif splitted_line[0] == 'f':
            face_vertex_ids = [int(c.split('/')[0]) for c in splitted_line[1:]]
            assert len(face_vertex_ids) == 3
            face_vertex_ids = [(ind - 1) if (ind >= 0) else (len(vs) + ind)
                               for ind in face_vertex_ids]
            faces.append(face_vertex_ids)
    f.close()
    vs = np.asarray(vs)
    faces = np.asarray(faces, dtype=int)
    assert np.logical_and(faces >= 0, faces < len(vs)).all()
    return torch.Tensor(vs), torch.Tensor(faces).to(int)

def calc_adj(faces):
    v1 = faces[:, 0]
    v2 = faces[:, 1]
    v3 = faces[:, 2]
    num_verts = int(faces.max())
    ### Caution !! ### Hardcoded here
    adj = torch.eye(252).to(faces.device)

    adj[(v1, v2)] = 1
    adj[(v1, v3)] = 1

    adj[(v2, v1)] = 1
    adj[(v2, v3)] = 1

    adj[(v3, v1)] = 1
    adj[(v3, v2)] = 1

        # normalizes symetric, binary adj matrix such that sum of each row is 1
    def normalize_adj(mx):
        rowsum = mx.sum(1)
        r_inv = (1./rowsum).view(-1)
        r_inv[r_inv != r_inv] = 0.
        mx = torch.mm(torch.eye(r_inv.shape[0]).to(mx.device)*r_inv, mx)
        return mx

    return normalize_adj(adj)


def remove_non_manifolds(mesh, faces):
    mesh.ve = [[] for _ in mesh.vs]
    edges_set = set()
    mask = torch.ones(len(faces), dtype=bool)
    _, face_areas = compute_face_normals_and_areas(mesh, faces)
    for face_id, face in enumerate(faces):
        if face_areas[face_id] == 0:
            mask[face_id] = False
            continue
        faces_edges = []
        is_manifold = False
        for i in range(3):
            cur_edge = (face[i], face[(i + 1) % 3])
            if cur_edge in edges_set:
                is_manifold = True
                break
            else:
                faces_edges.append(cur_edge)
        if is_manifold:
            mask[face_id] = False
        else:
            for idx, edge in enumerate(faces_edges):
                edges_set.add(edge)
    return faces[mask], face_areas[mask]


def build_gemm(mesh, faces, face_areas):
    """
    gemm_edges: array (#E x 4) of the 4 one-ring neighbors for each edge
    sides: array (#E x 4) indices (values of: 0,1,2,3) indicating where an edge is in the gemm_edge entry of the 4 neighboring edges
    for example edge i -> gemm_edges[gemm_edges[i], sides[i]] == [i, i, i, i]
    """
    mesh.ve = [[] for _ in mesh.vs]
    edge_nb = []
    sides = []
    edge2key = dict()
    edges = []
    edges_count = 0
    nb_count = []
    for face_id, face in enumerate(faces):
        faces_edges = []
        for i in range(3):
            cur_edge = (face[i], face[(i + 1) % 3])
            faces_edges.append(cur_edge)
        for idx, edge in enumerate(faces_edges):
            edge = tuple(sorted(list(edge)))
            faces_edges[idx] = edge
            if edge not in edge2key:
                edge2key[edge] = edges_count
                edges.append(list(edge))
                edge_nb.append([-1, -1, -1, -1])
                sides.append([-1, -1, -1, -1])
                mesh.ve[edge[0]].append(edges_count)
                mesh.ve[edge[1]].append(edges_count)
                mesh.edge_areas.append(0)
                nb_count.append(0)
                edges_count = edges_count + 1
            mesh.edge_areas[edge2key[edge]] = mesh.edge_areas[edge2key[edge]] + face_areas[face_id] / 3
        for idx, edge in enumerate(faces_edges):
            edge_key = edge2key[edge]
            edge_nb[edge_key][nb_count[edge_key]] = edge2key[faces_edges[(idx + 1) % 3]]
            edge_nb[edge_key][nb_count[edge_key] + 1] = edge2key[faces_edges[(idx + 2) % 3]]
            nb_count[edge_key] = nb_count[edge_key] + 2
        for idx, edge in enumerate(faces_edges):
            edge_key = edge2key[edge]
            sides[edge_key][nb_count[edge_key] - 2] = nb_count[edge2key[faces_edges[(idx + 1) % 3]]] - 1
            sides[edge_key][nb_count[edge_key] - 1] = nb_count[edge2key[faces_edges[(idx + 2) % 3]]] - 2

    mesh.edges = torch.Tensor(edges).to(torch.int32)
    mesh.gemm_edges = torch.Tensor(edge_nb).to(torch.int64)
    mesh.sides = torch.Tensor(sides).to(torch.int64)
    mesh.edges_count = edges_count
    mesh.edge_areas = torch.Tensor(mesh.edge_areas).to(torch.float32) / torch.sum(torch.Tensor(face_areas)) #todo whats the difference between edge_areas and edge_lenghts?


def compute_face_normals_and_areas(mesh, faces):
    face_normals = torch.cross(mesh.vs[faces[:, 1]] - mesh.vs[faces[:, 0]],
                            mesh.vs[faces[:, 2]] - mesh.vs[faces[:, 1]])
    face_areas = torch.sqrt((face_normals ** 2).sum(dim=1))
    face_normals = face_normals/face_areas.unsqueeze(-1)
    assert (not np.any((face_areas.unsqueeze(-1)).detach().numpy() == 0)), 'has zero area face: %s' % mesh.filename
    face_areas = face_areas*0.5
    return face_normals, face_areas


# Data augmentation methods
def augmentation(mesh, opt, faces=None):
    if hasattr(opt, 'scale_verts') and opt.scale_verts:
        scale_verts(mesh)
    if hasattr(opt, 'flip_edges') and opt.flip_edges:
        faces = flip_edges(mesh, opt.flip_edges, faces)
    return faces


def post_augmentation(mesh, opt):
    if hasattr(opt, 'slide_verts') and opt.slide_verts:
        slide_verts(mesh, opt.slide_verts)


def slide_verts(mesh, prct):
    edge_points = get_edge_points(mesh)
    dihedral = dihedral_angle(mesh, edge_points).squeeze() #todo make fixed_division epsilon=0
    thr = torch.mean(dihedral) + torch.std(dihedral)
    vids = torch.randperm(len(mesh.ve))
    target = int(prct * len(vids))
    shifted = 0
    for vi in vids:
        if shifted < target:
            edges = mesh.ve[vi]
            if min(dihedral[edges]) > 2.65:
                edge = mesh.edges[torch.randperm(edges)[0]]
                vi_t = edge[1] if vi == edge[0] else edge[0]
                nv = mesh.vs[vi] + torch.Tensor(np.random.uniform(0.2, 0.5)) * (mesh.vs[vi_t] - mesh.vs[vi])
                mesh.vs[vi] = nv
                shifted = shifted + 1
        else:
            break
    mesh.shifted = shifted / len(mesh.ve)


def scale_verts(mesh, mean=1, var=0.1):
    for i in range(mesh.vs.shape[1]):
        mesh.vs[:, i] = mesh.vs[:, i] * torch.Tensor(np.random.normal(mean, var))

def angles_from_faces(mesh, edge_faces, faces):
    normals = [None, None]
    for i in range(2):
        edge_a = mesh.vs[faces[edge_faces[:, i], 2]] - mesh.vs[faces[edge_faces[:, i], 1]]
        edge_b = mesh.vs[faces[edge_faces[:, i], 1]] - mesh.vs[faces[edge_faces[:, i], 0]]
        normals[i] = torch.cross(edge_a, edge_b)
        div = fixed_division(torch.norm(normals[i], dim=1), epsilon=0)
        normals[i] = normals[i]/div.unsqueeze(-1)
    dot = torch.sum(normals[0] * normals[1], dim=1).clamp(-1, 1)
    angles = torch.Tensor([np.pi]) - torch.acos(dot)
    return angles


def flip_edges(mesh, prct, faces):
    edge_count, edge_faces, edges_dict = get_edge_faces(faces)
    dihedral = angles_from_faces(mesh, edge_faces[:, 2:], faces)
    edges2flip = torch.randperm(edge_count)
    # print(dihedral.min())
    # print(dihedral.max())
    target = int(prct * edge_count)
    flipped = 0
    for edge_key in edges2flip:
        if flipped == target:
            break
        if dihedral[edge_key] > 2.7:
            edge_info = edge_faces[edge_key]
            if edge_info[3] == -1:
                continue
            new_edge = tuple(sorted(list(set(faces[edge_info[2]]) ^ set(faces[edge_info[3]]))))
            if new_edge in edges_dict:
                continue
            new_faces = torch.Tensor(
                [[edge_info[1], new_edge[0], new_edge[1]], [edge_info[0], new_edge[0], new_edge[1]]])
            if check_area(mesh, new_faces):
                del edges_dict[(edge_info[0], edge_info[1])]
                edge_info[:2] = [new_edge[0], new_edge[1]]
                edges_dict[new_edge] = edge_key
                rebuild_face(faces[edge_info[2]], new_faces[0])
                rebuild_face(faces[edge_info[3]], new_faces[1])
                for i, face_id in enumerate([edge_info[2], edge_info[3]]):
                    cur_face = faces[face_id]
                    for j in range(3):
                        cur_edge = tuple(sorted((cur_face[j], cur_face[(j + 1) % 3])))
                        if cur_edge != new_edge:
                            cur_edge_key = edges_dict[cur_edge]
                            for idx, face_nb in enumerate(
                                    [edge_faces[cur_edge_key, 2], edge_faces[cur_edge_key, 3]]):
                                if face_nb == edge_info[2 + (i + 1) % 2]:
                                    edge_faces[cur_edge_key, 2 + idx] = face_id
                flipped = flipped + 1
    # print(flipped)
    return faces


def rebuild_face(face, new_face):
    new_point = list(set(new_face) - set(face))[0]
    for i in range(3):
        if face[i] not in new_face:
            face[i] = new_point
            break
    return torch.Tensor(face)

def check_area(mesh, faces):
    face_normals = torch.cross(mesh.vs[faces[:, 1]] - mesh.vs[faces[:, 0]],
                            mesh.vs[faces[:, 2]] - mesh.vs[faces[:, 1]])
    face_areas = torch.sqrt((face_normals ** 2).sum(dim=1))
    face_areas = face_areas*0.5
    return face_areas[0] > 0 and face_areas[1] > 0


def get_edge_faces(faces):
    edge_count = 0
    edge_faces = []
    edge2keys = dict()
    for face_id, face in enumerate(faces):
        for i in range(3):
            cur_edge = tuple(sorted((face[i], face[(i + 1) % 3])))
            if cur_edge not in edge2keys:
                edge2keys[cur_edge] = edge_count
                edge_count = edge_count + 1
                edge_faces.append([cur_edge[0], cur_edge[1], -1, -1])
            edge_key = edge2keys[cur_edge]
            if edge_faces[edge_key][2] == -1:
                edge_faces[edge_key][2] = face_id
            else:
                edge_faces[edge_key][3] = face_id
    return edge_count, torch.Tensor(edge_faces).to(int), edge2keys


def set_edge_lengths(mesh, edge_points=None):
    if edge_points is not None:
        edge_points = get_edge_points(mesh)
    edge_lengths = torch.norm(mesh.vs[edge_points[:, 0]] - mesh.vs[edge_points[:, 1]], dim=1)
    mesh.edge_lengths = edge_lengths


def extract_features(mesh):
    features = []
    edge_points = get_edge_points(mesh)
    set_edge_lengths(mesh, edge_points)
    with np.errstate(divide='raise'):
        try:
            for extractor in [dihedral_angle, symmetric_opposite_angles, symmetric_ratios]:
                feature = extractor(mesh, edge_points)
                features.append(feature)
            return torch.cat(features, dim=0)
        except Exception as e:
            print(e)
            raise ValueError(mesh.filename, 'bad features')


def dihedral_angle(mesh, edge_points):
    normals_a = get_normals(mesh, edge_points, 0)
    normals_b = get_normals(mesh, edge_points, 3)
    dot = torch.sum(normals_a * normals_b, dim=1).clamp(-1, 1)
    angles = (torch.Tensor([np.pi]).to(edge_points.device) - torch.acos(dot)).unsqueeze(0)
    return angles


def symmetric_opposite_angles(mesh, edge_points):
    """ computes two angles: one for each face shared between the edge
        the angle is in each face opposite the edge
        sort handles order ambiguity
    """
    angles_a = get_opposite_angles(mesh, edge_points, 0)
    angles_b = get_opposite_angles(mesh, edge_points, 3)
    angles = torch.cat((angles_a.unsqueeze(0), angles_b.unsqueeze(0)), dim=0)
    angles = torch.sort(angles, dim=0)[0]
    return angles


def symmetric_ratios(mesh, edge_points):
    """ computes two ratios: one for each face shared between the edge
        the ratio is between the height / base (edge) of each triangle
        sort handles order ambiguity
    """
    ratios_a = get_ratios(mesh, edge_points, 0)
    ratios_b = get_ratios(mesh, edge_points, 3)
    ratios = torch.cat((ratios_a.unsqueeze(0), ratios_a.unsqueeze(0)), dim=0)
    return torch.sort(ratios, dim=0)[0]


def get_edge_points(mesh):
    """ returns: edge_points (#E x 4) tensor, with four vertex ids per edge
        for example: edge_points[edge_id, 0] and edge_points[edge_id, 1] are the two vertices which define edge_id 
        each adjacent face to edge_id has another vertex, which is edge_points[edge_id, 2] or edge_points[edge_id, 3]
    """
    edge_points = torch.zeros([mesh.edges_count, 4], dtype=torch.int32)
    for edge_id, edge in enumerate(mesh.edges):
        edge_points[edge_id] = get_side_points(mesh, edge_id)
        # edge_points[edge_id, 3:] = mesh.get_side_points(edge_id, 2)
    return edge_points.to(int)


def get_side_points(mesh, edge_id):
    # if mesh.gemm_edges[edge_id, side] == -1:
    #     return mesh.get_side_points(edge_id, ((side + 2) % 4))
    # else:
    edge_a = mesh.edges[edge_id]

    if mesh.gemm_edges[edge_id, 0] == -1:
        edge_b = mesh.edges[mesh.gemm_edges[edge_id, 2]]
        edge_c = mesh.edges[mesh.gemm_edges[edge_id, 3]]
    else:
        edge_b = mesh.edges[mesh.gemm_edges[edge_id, 0]]
        edge_c = mesh.edges[mesh.gemm_edges[edge_id, 1]]
    if mesh.gemm_edges[edge_id, 2] == -1:
        edge_d = mesh.edges[mesh.gemm_edges[edge_id, 0]]
        edge_e = mesh.edges[mesh.gemm_edges[edge_id, 1]]
    else:
        edge_d = mesh.edges[mesh.gemm_edges[edge_id, 2]]
        edge_e = mesh.edges[mesh.gemm_edges[edge_id, 3]]
    first_vertex = 0
    second_vertex = 0
    third_vertex = 0
    if edge_a[1] in edge_b:
        first_vertex = 1
    if edge_b[1] in edge_c:
        second_vertex = 1
    if edge_d[1] in edge_e:
        third_vertex = 1
    # print(torch.Tensor([edge_a[first_vertex], edge_a[1 - first_vertex], edge_b[second_vertex], edge_d[third_vertex]]))
    return torch.Tensor([edge_a[first_vertex], edge_a[1 - first_vertex], edge_b[second_vertex], edge_d[third_vertex]]).to(int)


def get_normals(mesh, edge_points, side):
    edge_a = mesh.vs[edge_points[:, side // 2 + 2]] - mesh.vs[edge_points[:, side // 2]]
    edge_b = mesh.vs[edge_points[:, 1 - side // 2]] - mesh.vs[edge_points[:, side // 2]]
    normals = torch.cross(edge_a, edge_b)
    div = fixed_division(torch.norm(normals, dim=1), epsilon=0.1)
    normals = normals/div.unsqueeze(-1)
    return normals

def get_opposite_angles(mesh, edge_points, side):
    edges_a = mesh.vs[edge_points[:, side // 2]] - mesh.vs[edge_points[:, side // 2 + 2]]
    edges_b = mesh.vs[edge_points[:, 1 - side // 2]] - mesh.vs[edge_points[:, side // 2 + 2]]

    edges_a = edges_a/fixed_division(torch.norm(edges_a, dim=1), epsilon=0.1).unsqueeze(-1)
    edges_b = edges_b/fixed_division(torch.norm(edges_b, dim=1), epsilon=0.1).unsqueeze(-1)
    dot = torch.sum(edges_a * edges_b, dim=1).clamp(-1, 1)
    return torch.acos(dot)


def get_ratios(mesh, edge_points, side):
    edges_lengths = torch.norm(mesh.vs[edge_points[:, side // 2]] - mesh.vs[edge_points[:, 1 - side // 2]], dim=1)
    point_o = mesh.vs[edge_points[:, side // 2 + 2]]
    point_a = mesh.vs[edge_points[:, side // 2]]
    point_b = mesh.vs[edge_points[:, 1 - side // 2]]
    line_ab = point_b - point_a
    projection_length = torch.sum(line_ab * (point_o - point_a), dim=1) / fixed_division(
        torch.norm(line_ab, dim=1), epsilon=0.1)
    closest_point = point_a + (projection_length / edges_lengths).unsqueeze(-1) * line_ab
    d = torch.norm(point_o - closest_point, dim=1)
    return d / edges_lengths

def fixed_division(to_div, epsilon):
    if epsilon == 0:
        to_div[to_div == 0] = 0.1
    else:
        to_div = to_div + epsilon
    return to_div
