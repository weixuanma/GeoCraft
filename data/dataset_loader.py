import os
import torch
import numpy as np
from torch.utils.data import Dataset
import open3d as o3d
from PIL import Image
import json

class ObjverseDataset(Dataset):
    def __init__(self, root_path, sample_num=110000, preprocess=True, transform=None):
        self.root_path = root_path
        self.sample_num = sample_num
        self.preprocess = preprocess
        self.transform = transform
        self.samples = self._load_samples()

    def _load_samples(self):
        sample_paths = [os.path.join(self.root_path, f) for f in os.listdir(self.root_path) if f.endswith(".obj")]
        if self.preprocess and len(sample_paths) > self.sample_num:
            sample_paths = sample_paths[:self.sample_num]
        return sample_paths

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        obj_path = self.samples[idx]
        mesh = o3d.io.read_triangle_mesh(obj_path)
        point_cloud = mesh.sample_points_uniformly(number_of_points=2048)
        pc_np = np.asarray(point_cloud.points, dtype=np.float32)
        
        img_path = obj_path.replace(".obj", ".png")
        img = Image.open(img_path).convert("RGB")
        
        if self.transform:
            pc_np = self.transform["point_cloud"](pc_np)
            img = self.transform["image"](img)
        
        return {
            "point_cloud": torch.tensor(pc_np),
            "image": img,
            "path": obj_path
        }

class GSODataset(Dataset):
    def __init__(self, root_path, sample_num=1000, transform=None):
        self.root_path = root_path
        self.sample_num = sample_num
        self.transform = transform
        self.samples = self._load_samples()

    def _load_samples(self):
        meta_path = os.path.join(self.root_path, "metadata.json")
        with open(meta_path, "r") as f:
            meta = json.load(f)
        sample_paths = [os.path.join(self.root_path, item["path"]) for item in meta[:self.sample_num]]
        return sample_paths

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        pc_path = self.samples[idx]
        pc = o3d.io.read_point_cloud(pc_path)
        pc_np = np.asarray(pc.points, dtype=np.float32)
        
        img_path = pc_path.replace(".ply", ".jpg")
        img = Image.open(img_path).convert("RGB")
        
        if self.transform:
            pc_np = self.transform["point_cloud"](pc_np)
            img = self.transform["image"](img)
        
        return {
            "point_cloud": torch.tensor(pc_np),
            "image": img,
            "path": pc_path
        }

class Pix3DDataset(Dataset):
    def __init__(self, root_path, sample_num=3000, transform=None):
        self.root_path = root_path
        self.sample_num = sample_num
        self.transform = transform
        self.samples = self._load_samples()

    def _load_samples(self):
        split_path = os.path.join(self.root_path, "test_split.txt")
        with open(split_path, "r") as f:
            sample_names = [line.strip() for line in f.readlines()[:self.sample_num]]
        sample_paths = [os.path.join(self.root_path, "models", name + ".obj") for name in sample_names]
        return sample_paths

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        obj_path = self.samples[idx]
        mesh = o3d.io.read_triangle_mesh(obj_path)
        point_cloud = mesh.sample_points_uniformly(number_of_points=2048)
        pc_np = np.asarray(point_cloud.points, dtype=np.float32)
        
        img_dir = os.path.join(self.root_path, "images", os.path.basename(obj_path)[:-4])
        img_path = os.path.join(img_dir, os.listdir(img_dir)[0])
        img = Image.open(img_path).convert("RGB")
        
        if self.transform:
            pc_np = self.transform["point_cloud"](pc_np)
            img = self.transform["image"](img)
        
        return {
            "point_cloud": torch.tensor(pc_np),
            "image": img,
            "path": obj_path
        }