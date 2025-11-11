import torch
import numpy as np
import open3d as o3d

def calculate_point_density(pc, k_neighbors=32, sigma=0.1):
    batch_size, num_points, _ = pc.shape
    device = pc.device
    
    dist_matrix = torch.cdist(pc, pc)
    k_dist = dist_matrix.topk(k=k_neighbors, dim=-1, largest=False)[0]
    
    density = torch.exp(-2 * sigma ** 2 / (k_dist ** 2 + 1e-6)).mean(dim=-1)
    return density

def compute_sdf(mesh, points):
    mesh_verts = torch.tensor(np.asarray(mesh.vertices), device=points.device, dtype=torch.float32)
    mesh_faces = torch.tensor(np.asarray(mesh.triangles), device=points.device, dtype=torch.long)
    
    batch_size, num_points, _ = points.shape
    sdf = torch.full((batch_size, num_points), float('inf'), device=points.device)
    
    for b in range(batch_size):
        point_batch = points[b]
        dist = torch.cdist(point_batch, mesh_verts)
        min_dist, _ = dist.min(dim=-1)
        
        inside = _point_in_mesh_batch(point_batch, mesh_verts, mesh_faces)
        sdf[b] = torch.where(inside, -min_dist, min_dist)
    
    return sdf

def _point_in_mesh_batch(points, verts, faces):
    num_points = points.shape[0]
    num_faces = faces.shape[0]
    
    v0 = verts[faces[:, 0]]
    v1 = verts[faces[:, 1]]
    v2 = verts[faces[:, 2]]
    
    ray_dir = torch.tensor([1.0, 0.0, 0.0], device=points.device).unsqueeze(0).expand(num_points, -1)
    points_expand = points.unsqueeze(1).expand(-1, num_faces, -1)
    
    edge1 = v1 - v0
    edge2 = v2 - v0
    norm = torch.cross(edge1, edge2, dim=-1)
    
    dir_dot_norm = torch.matmul(ray_dir.unsqueeze(1), norm.unsqueeze(-1)).squeeze(-1)
    mask = dir_dot_norm.abs() > 1e-6
    
    t = torch.matmul((v0 - points_expand).unsqueeze(2), norm.unsqueeze(-1)).squeeze(-1)
    t = t / (dir_dot_norm + 1e-6)
    t = torch.where(mask, t, torch.full_like(t, -1.0))
    
    p = points_expand + t.unsqueeze(-1) * ray_dir.unsqueeze(1)
    
    u = torch.matmul(torch.cross(ray_dir.unsqueeze(1), edge2.unsqueeze(0)), (p - v0).unsqueeze(-1)).squeeze(-1)
    u = u / (torch.matmul(norm.unsqueeze(0), norm.unsqueeze(-1)).squeeze(-1) + 1e-6)
    
    v = torch.matmul(torch.cross(edge1.unsqueeze(0), ray_dir.unsqueeze(1)), (p - v0).unsqueeze(-1)).squeeze(-1)
    v = v / (torch.matmul(norm.unsqueeze(0), norm.unsqueeze(-1)).squeeze(-1) + 1e-6)
    
    hit = (t > 1e-6) & (u >= 0) & (v >= 0) & (u + v <= 1) & mask
    count = hit.sum(dim=-1)
    return count % 2 == 1

def get_camera_projection_matrix(intrinsics, extrinsics):
    batch_size = intrinsics.shape[0]
    device = intrinsics.device
    
    K = torch.zeros(batch_size, 3, 3, device=device)
    K[:, 0, 0] = intrinsics[:, 0]
    K[:, 1, 1] = intrinsics[:, 1]
    K[:, 0, 2] = intrinsics[:, 2]
    K[:, 1, 2] = intrinsics[:, 3]
    K[:, 2, 2] = 1.0
    
    R = extrinsics[:, :3, :3]
    t = extrinsics[:, :3, 3].unsqueeze(-1)
    
    RT = torch.cat([R, t], dim=-1)
    P = torch.bmm(K, RT)
    return P