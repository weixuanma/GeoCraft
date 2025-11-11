import torch
import torch.nn as nn
import numpy as np
import open3d as o3d

class Voxelizer(nn.Module):
    def __init__(self, resolution=128, iso_surface_radius=0.02):
        super().__init__()
        self.resolution = resolution
        self.iso_surface_radius = iso_surface_radius

    def _compute_sdf(self, mesh, voxel_coords):
        mesh_verts = np.asarray(mesh.vertices)
        mesh_faces = np.asarray(mesh.triangles)
        
        sdf = []
        for coord in voxel_coords:
            dist = np.min(np.linalg.norm(mesh_verts - coord, axis=1))
            inside = self._point_in_mesh(coord, mesh_verts, mesh_faces)
            sdf_val = -dist if inside else dist
            sdf.append(sdf_val)
        return np.array(sdf)

    def _point_in_mesh(self, point, verts, faces):
        ray_dir = np.array([1.0, 0.0, 0.0])
        count = 0
        for face in faces:
            v0, v1, v2 = verts[face]
            if self._ray_triangle_intersect(point, ray_dir, v0, v1, v2):
                count += 1
        return count % 2 == 1

    def _ray_triangle_intersect(self, orig, dir, v0, v1, v2):
        edge1 = v1 - v0
        edge2 = v2 - v0
        norm = np.cross(edge1, edge2)
        if np.dot(norm, dir) == 0:
            return False
        
        t = np.dot(v0 - orig, norm) / np.dot(dir, norm)
        if t < 0:
            return False
        
        p = orig + t * dir
        u = np.dot(np.cross(dir, edge2), (p - v0)) / np.dot(norm, norm)
        v = np.dot(np.cross(edge1, dir), (p - v0)) / np.dot(norm, norm)
        return u >= 0 and v >= 0 and u + v <= 1

    def forward(self, mesh):
        device = next(self.parameters()).device if self.parameters() else torch.device("cpu")
        
        x = np.linspace(mesh.get_min_bound()[0], mesh.get_max_bound()[0], self.resolution)
        y = np.linspace(mesh.get_min_bound()[1], mesh.get_max_bound()[1], self.resolution)
        z = np.linspace(mesh.get_min_bound()[2], mesh.get_max_bound()[2], self.resolution)
        grid_x, grid_y, grid_z = np.meshgrid(x, y, z, indexing="ij")
        voxel_coords = np.stack([grid_x, grid_y, grid_z], dim=-1).reshape(-1, 3)
        
        sdf = self._compute_sdf(mesh, voxel_coords)
        sdf_grid = sdf.reshape(self.resolution, self.resolution, self.resolution)
        
        sparse_mask = np.abs(sdf_grid) <= self.iso_surface_radius
        sparse_voxels = torch.tensor(sdf_grid, device=device).unsqueeze(0).unsqueeze(0)
        sparse_voxels[~sparse_mask] = 0.0
        return sparse_voxels