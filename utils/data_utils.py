import os
import xml.etree.ElementTree as ET
import torch
from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms

class ConcreteDefectDataset(Dataset):
    def __init__(self, data_dir, split="train", image_size=640, augment=False):
        self.data_dir = os.path.join(data_dir, split)
        self.image_size = image_size
        self.augment = augment
        self.image_paths = self._collect_image_paths()
        self.annotations = self._load_annotations()
        self.transform = self._build_transform_pipeline()

    def _collect_image_paths(self):
        img_extensions = (".jpg", ".jpeg", ".png")
        paths = []
        for root, _, files in os.walk(self.data_dir):
            for file in files:
                if file.lower().endswith(img_extensions):
                    paths.append(os.path.join(root, file))
        return paths

    def _load_annotations(self):
        annotations = []
        for img_path in self.image_paths:
            anno_path = img_path.replace(os.path.splitext(img_path)[1], ".xml")
            if not os.path.exists(anno_path):
                annotations.append({"bboxes": [], "cls_ids": []})
                continue
            
            tree = ET.parse(anno_path)
            root = tree.getroot()
            bboxes = []
            cls_ids = []
            
            for obj in root.findall("object"):
                cls_name = obj.find("name").text
                if cls_name.lower() in ["crack"]:
                    bndbox = obj.find("bndbox")
                    x1 = float(bndbox.find("xmin").text)
                    y1 = float(bndbox.find("ymin").text)
                    x2 = float(bndbox.find("xmax").text)
                    y2 = float(bndbox.find("ymax").text)
                    bboxes.append([x1, y1, x2, y2])
                    cls_ids.append(0)
            
            annotations.append({"bboxes": bboxes, "cls_ids": cls_ids})
        return annotations

    def _build_transform_pipeline(self):
        transform_list = [
            transforms.Resize((self.image_size, self.image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ]
        if self.augment:
            transform_list.insert(0, transforms.RandomHorizontalFlip(p=0.5))
            transform_list.insert(0, transforms.RandomVerticalFlip(p=0.3))
            transform_list.insert(0, transforms.ColorJitter(brightness=0.2, contrast=0.2))
        return transforms.Compose(transform_list)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert("RGB")
        image = self.transform(image)
        
        anno = self.annotations[idx]
        bboxes = torch.tensor(anno["bboxes"], dtype=torch.float32) if anno["bboxes"] else torch.empty((0, 4), dtype=torch.float32)
        cls_ids = torch.tensor(anno["cls_ids"], dtype=torch.long) if anno["cls_ids"] else torch.empty(0, dtype=torch.long)
        
        return image, bboxes, cls_ids

    def __len__(self):
        return len(self.image_paths)