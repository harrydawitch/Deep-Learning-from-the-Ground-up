import os
import torch 
import torchvision
import torch.nn as nn
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
import numpy as np
import math
from PIL import Image


class ButterflyDataset(Dataset):
    def __init__(self, root_dir, transform= None):
        self.root_dir = root_dir
        self.transform= transform
        self.train_images = os.listdir(os.path.join(self.root_dir, "images"))
        self.labels_images = os.listdir(os.path.join(self.root_dir, "segmentations"))

        
    def __getitem__(self, idx):
        train_img_path = os.path.join(self.root_dir, "images", self.train_images[idx])
        label_img_path = os.path.join(self.root_dir, "segmentations", self.labels_images[idx])
        
        train_img = np.array(Image.open(train_img_path).convert("RGB"))
        label_img = np.array(Image.open(label_img_path).convert("L"), dype= np.float32)
        label_img[label_img==255.0] = 1.0
        
        if self.transform:
            augmentation = self.transform(image= train_img, mask= label_img)
            train_img = augmentation['train_img']
            label_img = augmentation['label_img']
            
        return train_img, label_img

    def __len__(self):
        return len(self.train_images)
    

