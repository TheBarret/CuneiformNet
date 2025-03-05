import os
import json
import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim

from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image

import logging
from datetime import datetime
import yaml

# Dataset Settings
DATASET_SHAPE = (64, 64)

# Configure logging
logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(levelname)s: %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# Configuration Management
class ConfigManager:
    def __init__(self, config_path='/content/config.yaml'):
        with open(config_path, 'r') as file:
            self.config = yaml.safe_load(file)
        
        torch.manual_seed(self.config['seed'])
        np.random.seed(self.config['seed'])

    def get(self, key):
        return self.config.get(key)

# Custom Dataset for Cuneiform Images
class CuneiformDataset(Dataset):
    def __init__(self, image_dir, metadata_path, transform=None):
        self.image_dir = image_dir
        self.transform = transform or transforms.Compose([
            transforms.Resize(DATASET_SHAPE),
            transforms.Grayscale(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5])
        ])
        
        with open(metadata_path, 'r') as f:
            self.metadata = json.load(f)
        
        self.image_paths, self.labels = [], []
        for filename, info in self.metadata.items():
            full_path = os.path.join(image_dir, filename)
            if os.path.exists(full_path):
                label_number = int(filename.split('_')[1].split('.')[0], 16)
                label_binary = [int(bit) for bit in f'{label_number:032b}']
                
                self.image_paths.append(full_path)
                self.labels.append(torch.tensor(label_binary, dtype=torch.float32))
        
        logger.info(f"Loaded {len(self.image_paths)} images")

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image = Image.open(self.image_paths[idx]).convert('L')
        image = self.transform(image)
        return image, self.labels[idx]

class CuneiformNet(nn.Module):
    def __init__(self, num_classes=32):
        super(CuneiformNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2)
        )
        
        self.classifier = nn.Sequential(
            nn.Linear(256 * 8 * 8, 1024),
            nn.ReLU(inplace=True),
            nn.Dropout(0.6),
            nn.Linear(1024, num_classes),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        return self.classifier(x)

def train_model(config_manager):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
    dataset = CuneiformDataset(
        image_dir="/content/images", 
        metadata_path="/content/images/cuneiform.json"
    )
    
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    
    # Adjust batch size for T4
    batch_size = min(64, config_manager.get('batch_size'))
    num_workers = 2
    
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, 
        num_workers=num_workers, pin_memory=True)
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, 
        num_workers=num_workers, pin_memory=True)
    
    model = CuneiformNet().to(device)
    criterion = nn.BCELoss()
    #optimizer = optim.Adam(model.parameters(), lr=config_manager.get('learning_rate'))
    optimizer = optim.Adam(model.parameters(), lr=config_manager.get('learning_rate'), weight_decay=1e-4)
    #scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, T_max=config_manager.get('epochs'))
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, 
        mode='min',        # Lower validation loss is better
        factor=0.5,        # Reduce LR by half
        patience=5,        # Wait 5 epochs before reducing LR
        threshold=0.01,    # Minimum change to count as improvement
    )
    num_epochs = config_manager.get('epochs')
    best_val_loss = float('inf')
    
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
        
        val_loss = 0.0
        model.eval()
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                val_loss += criterion(outputs, labels).item()
        
        train_loss /= len(train_loader)
        val_loss /= len(val_loader)
        scheduler.step(val_loss)
        
        logger.info(f'Epoch [{epoch+1}/{num_epochs}] Train Loss: {train_loss:.4f} Val Loss: {val_loss:.4f}')
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), '/content/cuneiform_model_latest.pth')
            logger.info('Model saved.')
    
    logger.info("Training completed successfully!")

def main():
    config_manager = ConfigManager()
    train_model(config_manager)

if __name__ == "__main__":
    main()
