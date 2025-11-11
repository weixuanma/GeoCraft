import torch
import numpy as np
import open3d as o3d

class MeshGenerator:
    def __init__(self, vertex_offset_range=(-0.1, 0.1), face_type="triangle"):
        self.vertex_offset_range = vertex_offset_range
        self.face_type = face_type

    def decode_tokens(self, tokens, point_feat):
        batch_size = tokens.shape[0]
        vertices_list = []
        faces_list = []
        
        for b in range(batch_size):
            tokens_b = tokens[b]
            vertices = []
            faces = []
            prev_vertex = torch.zeros(3, device=tokens_b.device)
            
            for token in tokens_b:
                if token < 3:
                    offset = torch.FloatTensor(self.vertex_offset_range).uniform_(size=(3,)).to(tokens_b.device)
                    new_vertex = prev_vertex + offset
                    vertices.append(new_vertex)
                    prev_vertex = new_vertex
                else:
                    if len(vertices) >= 3:
                        if self.face_type == "triangle":
                            face = [len(vertices)-3, len(vertices)-2, len(vertices)-1]
                            faces.append(face)
            
            vertices = torch.stack(vertices) if vertices else torch.empty(0, 3)
            faces = torch.tensor(faces, dtype=torch.int64) if faces else torch.empty(0, 3)
            vertices_list.append(vertices)
            faces_list.append(faces)
        return vertices_list, faces_list

    def generate_mesh(self, vertices, faces):
        mesh = o3d.geometry.TriangleMesh()
        mesh.vertices = o3d.utility.Vector3dVector(vertices.cpu().detach().numpy())
        if faces.shape[0] > 0:
            mesh.triangles = o3d.utility.Vector3iVector(faces.cpu().detach().numpy())
        mesh.compute_vertex_normals()
        return mesh

    def post_process(self, mesh, smooth_iterations=5):
        mesh = mesh.filter_smooth_simple(number_of_iterations=smooth_iterations)
        mesh.remove_duplicated_vertices()
        mesh.remove_duplicated_triangles()
        mesh.remove_degenerate_triangles()
        return mesh