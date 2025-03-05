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

# Settings
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
    def __init__(self, config_path='config.yaml'):
        with open(config_path, 'r') as file:
            self.config = yaml.safe_load(file)
        
        # Ensure reproducibility
        torch.manual_seed(self.config['seed'])
        np.random.seed(self.config['seed'])

    def get(self, key):
        return self.config.get(key)

def load_json_with_encoding(file_path):
    encodings = [
        'utf-8',      # Most common modern encoding
        'latin-1',    # Fallback encoding that can read most byte sequences
        'cp1252',     # Windows default encoding
        'utf-16',     # Unicode encoding
        'utf-8-sig'   # UTF-8 with Byte Order Mark
    ]
    for encoding in encodings:
        try:
            with open(file_path, 'r', encoding=encoding) as f:
                print(f'Opening JSON [encoding: {encoding}]')
                return json.load(f)
        except (UnicodeDecodeError, json.JSONDecodeError) as e:
            logger.warning(f"Failed to load JSON with {encoding} encoding: {e}")
    raise ValueError(f"Could not load JSON file {file_path} with any known encoding")

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
        
        # Load metadata
        #with open(metadata_path, 'r') as f:
        #    self.metadata = json.load(f)
        self.metadata = load_json_with_encoding(metadata_path)
        
        # Prepare image paths and labels
        self.image_paths = []
        self.labels = []
        
        for filename, info in self.metadata.items():
            full_path = os.path.join(image_dir, filename)
            if os.path.exists(full_path):
                # Extract hexadecimal number from filename
                label_number = int(filename.split('_')[1].split('.')[0], 16)

                # Convert to binary representation
                label_binary = [int(bit) for bit in f'{label_number:032b}']
                
                self.image_paths.append(full_path)
                self.labels.append(torch.tensor(label_binary, dtype=torch.float32))
        
        logger.info(f"Loaded {len(self.image_paths)} images")

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image = Image.open(self.image_paths[idx]).convert('L')  # Grayscale
        image = self.transform(image)
        return image, self.labels[idx]

class CuneiformNet(nn.Module):
    def __init__(self, num_classes=32):
        super(CuneiformNet, self).__init__()
        # Multi-scale feature extraction
        self.multi_scale_features = nn.ModuleList([
            # Different kernel sizes to capture varied line thicknesses
            nn.Sequential(
                nn.Conv2d(1, 32, kernel_size=1, padding=0),
                nn.BatchNorm2d(32),
                nn.ReLU(inplace=True)
            ),
            nn.Sequential(
                nn.Conv2d(1, 32, kernel_size=3, padding=1),
                nn.BatchNorm2d(32),
                nn.ReLU(inplace=True)
            ),
            nn.Sequential(
                nn.Conv2d(1, 32, kernel_size=5, padding=2),
                nn.BatchNorm2d(32),
                nn.ReLU(inplace=True)
            )
        ])
        
        # Attention mechanism to focus on important features
        self.feature_attention = nn.Sequential(
            nn.Conv2d(96, 96, kernel_size=1),
            nn.BatchNorm2d(96),
            nn.Sigmoid()
        )
        
        # Detailed feature extraction
        self.detailed_features = nn.Sequential(
            # Deeper and more complex feature extraction
            nn.Conv2d(96, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            # Dilated convolutions to capture wider context
            nn.Conv2d(256, 512, kernel_size=3, padding=2, dilation=2),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True)
        )
        
        # Advanced classifier with more complexity
        self.classifier = nn.Sequential(
            nn.Linear(512 * 16 * 16, 2048),  # Changed from 512 * 8 * 8
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.BatchNorm1d(2048),
            
            nn.Linear(2048, 1024),
            nn.ReLU(inplace=True),
            nn.Dropout(0.4),
            nn.BatchNorm1d(1024),
            
            nn.Linear(1024, num_classes),
            nn.Sigmoid()
        )

    def forward(self, x):
        # Multi-scale feature extraction
        multi_scale_features = [scale(x) for scale in self.multi_scale_features]
        
        # Concatenate multi-scale features
        combined_features = torch.cat(multi_scale_features, dim=1)
        
        # Apply attention mechanism
        attention_map = self.feature_attention(combined_features)
        focused_features = combined_features * attention_map
        
        # Further feature extraction
        detailed_features = self.detailed_features(focused_features)
        
        # Flatten and classify
        flattened = torch.flatten(detailed_features, 1)
        return self.classifier(flattened)

def train_model(config_manager):
    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")

    # Dataset and DataLoader
    dataset = CuneiformDataset(
        image_dir=".\\images\\", 
        metadata_path=".\\images\\cuneiform.json"
    )
    
    # Split dataset
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(
        train_dataset, 
        batch_size=config_manager.get('batch_size'), 
        shuffle=True, 
        num_workers=4
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=config_manager.get('batch_size'), 
        shuffle=False, 
        num_workers=4
    )

    # Model, Loss, Optimizer
    model = CuneiformNet().to(device)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(
        model.parameters(), 
        lr=config_manager.get('learning_rate')
    )
    
    # Learning Rate Scheduler
    # scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        # optimizer, 
        # mode='min', 
        # factor=0.1, 
        # patience=5
    # )
    

    # Training Loop
    num_epochs = config_manager.get('epochs')
    best_val_loss = float('inf')
    
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)
    
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

        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                val_loss += criterion(outputs, labels).item()

        train_loss /= len(train_loader)
        val_loss /= len(val_loader)

        logger.info(f'Epoch [{epoch+1}/{num_epochs}], '
                    f'Train Loss: {train_loss:.4f}, '
                    f'Val Loss: {val_loss:.4f}')

        scheduler.step(val_loss)

        # Model Checkpointing
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            model_save_path = f'cuneiform_model_{timestamp}.pth'
            torch.save({
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'epoch': epoch,
                'loss': best_val_loss
            }, model_save_path)
            logger.info(f'Model saved to {model_save_path}')

    logger.info("Training completed successfully!")

def main():
    config_manager = ConfigManager()
    train_model(config_manager)

if __name__ == "__main__":
    main()