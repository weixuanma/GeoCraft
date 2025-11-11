import numpy as np
import torch
from PIL import Image
import random
import cv2

class PointCloudAugmenter:
    def __init__(self, rotate_range=(-15, 15), translate_range=(-0.1, 0.1), scale_range=(0.9, 1.1)):
        self.rotate_range = rotate_range
        self.translate_range = translate_range
        self.scale_range = scale_range

    def rotate(self, pc):
        angle = random.uniform(*self.rotate_range) * np.pi / 180
        rot_mat = np.array([
            [np.cos(angle), -np.sin(angle), 0],
            [np.sin(angle), np.cos(angle), 0],
            [0, 0, 1]
        ])
        return np.dot(pc, rot_mat.T)

    def translate(self, pc):
        trans = np.array([
            random.uniform(*self.translate_range),
            random.uniform(*self.translate_range),
            random.uniform(*self.translate_range)
        ])
        return pc + trans

    def scale(self, pc):
        scale = random.uniform(*self.scale_range)
        return pc * scale

    def __call__(self, pc):
        pc = self.rotate(pc)
        pc = self.translate(pc)
        pc = self.scale(pc)
        return pc

class ImageAugmenter:
    def __init__(self, flip_prob=0.5, brightness_range=(0.8, 1.2), contrast_range=(0.8, 1.2), resize_size=(256, 256)):
        self.flip_prob = flip_prob
        self.brightness_range = brightness_range
        self.contrast_range = contrast_range
        self.resize_size = resize_size

    def flip(self, img):
        if random.random() < self.flip_prob:
            return img.transpose(Image.FLIP_LEFT_RIGHT)
        return img

    def adjust_brightness(self, img):
        factor = random.uniform(*self.brightness_range)
        return Image.fromarray(np.clip(np.array(img) * factor, 0, 255).astype(np.uint8))

    def adjust_contrast(self, img):
        factor = random.uniform(*self.contrast_range)
        gray = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2GRAY)
        mean = np.mean(gray)
        return Image.fromarray(np.clip((np.array(img) - mean) * factor + mean, 0, 255).astype(np.uint8))

    def resize(self, img):
        return img.resize(self.resize_size)

    def __call__(self, img):
        img = self.resize(img)
        img = self.flip(img)
        img = self.adjust_brightness(img)
        img = self.adjust_contrast(img)
        return img