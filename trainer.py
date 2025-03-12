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
from torch.amp import autocast, GradScaler  # Updated for mixed precision training
from sklearn.utils.class_weight import compute_class_weight

# Dataset Settings
DATASET_SHAPE = (256, 256)

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
        # self.transform = transform or transforms.Compose([
            # transforms.Resize(DATASET_SHAPE),
            # transforms.Grayscale(),
            # transforms.RandomHorizontalFlip(),  # Data augmentation
            # transforms.RandomRotation(10),     # Data augmentation
            # transforms.ToTensor(),
            # transforms.Normalize(mean=[0.5], std=[0.5]),
            # transforms.Lambda(lambda x: x + torch.randn_like(x) * 0.05)  # Add noise
        # ])
        
        # better diversity data aug
        self.transform = transforms.Compose([
            transforms.Resize(DATASET_SHAPE),
            transforms.Grayscale(),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
            # Random translation
            transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5]),
            # Add noise
            transforms.Lambda(lambda x: x + torch.randn_like(x) * 0.05)
        ])
        
        with open(metadata_path, 'r') as f:
            self.metadata = json.load(f)
        
        self.image_paths, self.labels = [], []
        for filename, info in self.metadata.items():
            full_path = os.path.join(image_dir, filename)
            if os.path.exists(full_path):
                label_number = int(filename.split('_')[1].split('.')[0], 16) - 0x12000
                # Ensure labels are in [0, 1023]
                label_number = max(0, min(1023, label_number))  
                
                self.labels.append(torch.tensor(label_number, dtype=torch.long))
                self.image_paths.append(full_path)
        
        labels = torch.tensor(self.labels, dtype=torch.long)
        logger.info(f"Images: {len(self.image_paths)}")
        logger.info(f"Labels: min {labels.min()} | max: {labels.max()}")

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image = Image.open(self.image_paths[idx]).convert('L')
        image = self.transform(image)
        return image, self.labels[idx]

# Enhanced Model Architecture
class CuneiformNet(nn.Module):
    def __init__(self, num_classes=1024):
        super(CuneiformNet, self).__init__()
        
        self.features = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, stride=2, padding=1),
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
            nn.MaxPool2d(2, 2),
            
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2)
        )
        
        # Calculate the correct input size for the Linear layer
        self.classifier_input_size = 512 * (DATASET_SHAPE[0] // 32) * (DATASET_SHAPE[1] // 32)  # Adjusted for pooling layers
        
        # self.classifier = nn.Sequential(
            # nn.Linear(self.classifier_input_size, 1024),
            # nn.ReLU(inplace=True),
            # nn.Dropout(0.5),
            # nn.Linear(1024, num_classes)
        # )
        
        # increase neurons
        self.classifier = nn.Sequential(
            nn.Linear(self.classifier_input_size, 2048),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            # Additional layer for more capacity
            nn.Linear(2048, 1024),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(1024, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)  # Flatten all dimensions except batch
        x = self.classifier(x)
        return x

# Training Function
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
    
    batch_size = 32  # Adjusted for T4 GPU
    num_workers = 2  # Adjusted to avoid warnings
    
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, 
        num_workers=num_workers, pin_memory=True)
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, 
        num_workers=num_workers, pin_memory=True)
    
    model = CuneiformNet(num_classes=1024).to(device)
        
    # Compute class weights
    labels = [label.item() for label in dataset.labels]  # Convert tensor to list
    class_weights = compute_class_weight('balanced', classes=np.unique(labels), y=labels)
    class_weights = torch.tensor(class_weights, dtype=torch.float32).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
   
    #optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
    #scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config_manager.get('epochs'))
    
    optimizer = optim.Adam(model.parameters(), lr=0.0001, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5)
   
    # Mixed precision training
    scaler = GradScaler()
    
    num_epochs = config_manager.get('epochs')
    best_val_loss = float('inf')
    early_stopping_patience = 20
    patience_counter = 0
    
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()
            with autocast(device_type='cuda', dtype=torch.float16):  # Mixed precision
                outputs = model(images)
                loss = criterion(outputs, labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
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
            logger.info(f'Model saved with validation loss: {best_val_loss:.4f}')
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= early_stopping_patience:
                logger.info('Early stopping triggered.')
                break
    
    logger.info("Training completed successfully!")

# Main Function
def main():
    config_manager = ConfigManager()
    train_model(config_manager)

if __name__ == "__main__":
    main()
