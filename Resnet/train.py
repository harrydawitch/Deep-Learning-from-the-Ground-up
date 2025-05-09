import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from model import resnet34
from dataset import FashionDataset

BATCH_SIZE = 32
EPOCHS = 20
LEARNING_RATE = 1e-4
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
data_path = r"Dataset"

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std= [0.5, 0.5, 0.5])
])


fashion_dataset = FashionDataset(root_dir = data_path, transforms= transform) 
dataloader = DataLoader(fashion_dataset, batch_size= BATCH_SIZE, shuffle= True)


model = resnet34().to(device= DEVICE)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr= LEARNING_RATE)

for epoch in range(EPOCHS):
    for image, label in dataloader:
        image, label= image.to(device= DEVICE), label.to(device= DEVICE)
        optimizer.zero_grad()
        preds = model(image)
        loss = criterion(preds, label)
        loss.backward()
        
        optimizer.step()