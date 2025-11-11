import os
import numpy as np
import open3d as o3d
from PIL import Image
import cv2
from skimage.metrics import structural_similarity as ssim

class DataFilter:
    def __init__(self, root_path, output_path, sample_num=110000):
        self.root_path = root_path
        self.output_path = output_path
        self.sample_num = sample_num
        os.makedirs(self.output_path, exist_ok=True)

    def filter_objverse(self):
        obj_paths = [os.path.join(self.root_path, f) for f in os.listdir(self.root_path) if f.endswith(".obj")]
        selected = []
        
        for path in obj_paths:
            if len(selected) >= self.sample_num:
                break
            mesh = o3d.io.read_triangle_mesh(path)
            if not mesh.is_empty() and len(mesh.vertices) > 100:
                selected.append(path)
        
        for path in selected:
            dest = os.path.join(self.output_path, os.path.basename(path))
            mesh = o3d.io.read_triangle_mesh(path)
            o3d.io.write_triangle_mesh(dest, mesh)
            
            img_src = path.replace(".obj", ".png")
            img_dest = os.path.join(self.output_path, os.path.basename(img_src))
            img = Image.open(img_src)
            img.save(img_dest)

class MeshValidator:
    def __init__(self, ssim_threshold=0.85, euler_char_threshold=2, normal_consistency=0.9, patch_variance=0.1):
        self.ssim_threshold = ssim_threshold
        self.euler_char_threshold = euler_char_threshold
        self.normal_consistency = normal_consistency
        self.patch_variance = patch_variance

    def calculate_euler_char(self, mesh):
        v = len(mesh.vertices)
        e = len(mesh.edges)
        f = len(mesh.triangles)
        return v - e + f

    def check_normal_consistency(self, mesh):
        normals = np.asarray(mesh.vertex_normals)
        dot_products = np.mean([np.dot(normals[i], normals[j]) for i, j in mesh.edges], axis=0)
        return np.mean(dot_products) >= self.normal_consistency

    def calculate_patch_variance(self, mesh):
        areas = np.asarray(mesh.triangle_area_per_vertex()).mean(axis=1)
        return np.var(areas) <= self.patch_variance

    def validate_mesh(self, mesh, ref_image, mesh_render):
        euler_char = self.calculate_euler_char(mesh)
        if euler_char > self.euler_char_threshold:
            return False
        
        if not self.check_normal_consistency(mesh):
            return False
        
        if not self.calculate_patch_variance(mesh):
            return False
        
        ref_gray = cv2.cvtColor(np.array(ref_image), cv2.COLOR_RGB2GRAY)
        render_gray = cv2.cvtColor(np.array(mesh_render), cv2.COLOR_RGB2GRAY)
        ssim_val = ssim(ref_gray, render_gray)
        return ssim_val >= self.ssim_threshold