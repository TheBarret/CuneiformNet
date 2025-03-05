import os
import json
import torch
import random
import numpy as np
from PIL import Image
import torch.nn as nn
import torchvision.transforms as transforms
import logging
from typing import List, Tuple

# Configure logging
logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(levelname)s: %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# Reuse the same neural network architecture from the training script
class CuneiformNet(nn.Module):
    def __init__(self, num_classes=32):
        super(CuneiformNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        
        self.classifier = nn.Sequential(
            nn.Linear(128 * 4 * 4, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return torch.sigmoid(x)

class CuneiformPredictor:
    def __init__(self, model_path: str, metadata_path: str):
        # Load metadata
        with open(metadata_path, 'r', encoding='utf-8') as f:
            self.metadata = json.load(f)
        
        # Device configuration
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Image transformation
        self.transform = transforms.Compose([
            transforms.Resize((32, 32)),
            transforms.Grayscale(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5])
        ])
        
        # Load the model
        self.model = CuneiformNet().to(self.device)
        checkpoint = torch.load(model_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()
        
        logger.info(f"Model loaded from {model_path}")

    def predict(self, image_path: str, top_k: int = 5) -> List[Tuple[str, float, str]]:
        # Load and transform image
        image = Image.open(image_path)
        input_tensor = self.transform(image).unsqueeze(0).to(self.device)
        
        # Predict
        with torch.no_grad():
            output = self.model(input_tensor).cpu().numpy()[0]
        
        # Convert binary output to decimal
        predicted_decimal = int(''.join([str(int(round(bit))) for bit in output]), 2)
        
        # Find matching metadata entries
        candidates = []
        for filename, info in self.metadata.items():
            # Convert unicode to decimal
            unicode_decimal = int(info['unicode'].replace('U+', ''), 16)
            
            # Calculate confidence (similarity between prediction and actual)
            confidence = 1 - np.mean(np.abs(np.array(output) - np.array([int(bit) for bit in f'{unicode_decimal:032b}'])))
            
            candidates.append((
                info['unicode'], 
                float(confidence), 
                info['description']
            ))
        
        # Sort candidates by confidence
        candidates.sort(key=lambda x: x[1], reverse=True)
        
        return candidates[:top_k]

def main():
    MODEL_PATH = './cuneiform_model_latest.pth'
    METADATA_PATH = './images/cuneiform.json'
    #IMAGE_PATH = './images/cuneiform_1230E.png'
    IMAGE_PATH = random.choice([os.path.join('./images/', f) for f in os.listdir('./images/') if f.endswith('.png') and f.startswith('cuneiform_')])
    
    logger.info(f"Specimen: {IMAGE_PATH}")
    
    # Create predictor
    predictor = CuneiformPredictor(MODEL_PATH, METADATA_PATH)
    
    # Get predictions
    predictions = predictor.predict(IMAGE_PATH)
    
    # Display results
    print("\nTop Predictions:")
    for i, (unicode, confidence, description) in enumerate(predictions, 1):
        print(f"{i}. Unicode: {unicode}")
        print(f"   Confidence: {confidence:.2%}")
        print(f"   Description: {description}\n")

if __name__ == "__main__":
    main()