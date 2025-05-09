import torch
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transfroms
from PIL import Image
import os


class FashionDataset(Dataset):
    def __init__(self, root_dir, transforms= None):
        
        self.root_dir = root_dir
        self.transforms= transforms
        
        self.images = []
        self.labels = []
        
        self.classes = sorted(os.listdir(self.root_dir))
        self.classes_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}
        
        for cls in self.classes:
            class_path = os.path.join(self.root_dir, cls)
            for image_name in os.listdir(class_path):
                image_path = os.join.path(class_path, image_name)
                
                self.images.append(image_path)
                self.labels.append(self.classes_to_idx[cls])

    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        img_path = self.images[idx]
        label = self.labels[idx]
        
        image = Image.open(img_path).convert("RGB")
        
        if self.transforms:
            image = self.transforms(image)
            
        return image, label