import os
import json
import torch
import random
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import torch.nn as nn
import torchvision.transforms as transforms
import logging
from typing import List, Tuple

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

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

class CuneiformPredictor:
    def __init__(self, model_path: str, metadata_path: str, truth_table_path: str, device: str = "auto"):
        with open(metadata_path, "r", encoding="utf-8") as f:
            self.metadata = json.load(f)
        
        self.truth_table = self.load_truth_table(truth_table_path)
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu") if device == "auto" else torch.device(device)
        logger.info(f"Using device: {self.device}")
        
        self.transform = transforms.Compose([
            transforms.Resize((64, 64)),
            transforms.Grayscale(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5])
        ])
        
        self.model = CuneiformNet().to(self.device)
        self.load_model(model_path)

    def load_model(self, model_path: str):
        try:
            checkpoint = torch.load(model_path, map_location=self.device)
            self.model.load_state_dict(checkpoint["model_state_dict"] if "model_state_dict" in checkpoint else checkpoint)
            self.model.eval()
            logger.info(f"Model loaded from {model_path}")
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            raise RuntimeError("Failed to load model.")

    def load_truth_table(self, path: str):
        truth_data = {}
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split("=")
                if len(parts) == 2:
                    filename = parts[0]
                    # Extract the numeric part of the filename
                    key = int(''.join(filter(str.isdigit, filename)))
                    truth_data[key] = parts[1].strip()
        return truth_data

    def predict(self, image_path: str, top_k: int = 5) -> List[Tuple[str, float, str]]:
        try:
            image = Image.open(image_path).convert("L")
        except Exception as e:
            logger.error(f"Error loading image: {e}")
            return []
        
        input_tensor = self.transform(image).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            output = self.model(input_tensor).cpu().numpy()[0]
        
        predicted_decimal = int("".join([str(int(round(bit))) for bit in output]), 2)
        candidates = []
        
        for filename, info in self.metadata.items():
            unicode_decimal = int(info["unicode"].replace("U+", ""), 16)
            confidence = 1 - np.mean(np.abs(output - np.array([int(bit) for bit in f"{unicode_decimal:032b}"])))
            candidates.append((info["unicode"], float(confidence), info["description"]))
        
        candidates.sort(key=lambda x: x[1], reverse=True)
        return candidates[:top_k]

    def visualize_results(self, image_path: str, predictions: List[Tuple[str, float, str]]):
        fig, axes = plt.subplots(1, len(predictions) + 1, figsize=(24, 8))
        input_img = Image.open(image_path)
        axes[0].imshow(input_img, cmap='gray')
        axes[0].set_title("Input Image")
        axes[0].axis("off")
        
        for i, (unicode_char, confidence, desc) in enumerate(predictions):
            hex_code = unicode_char.replace("U+", "").lower()
            pred_img_path = f"./images/cuneiform_{hex_code}.png"
            
            if os.path.exists(pred_img_path):
                pred_img = Image.open(pred_img_path)
            else:
                pred_img = Image.new("L", (64, 64), color=255)
            
            axes[i + 1].imshow(pred_img, cmap='gray')
            axes[i + 1].set_title(f"{unicode_char}\n{desc}\n({confidence:.2%})")
            axes[i + 1].axis("off")
        
        plt.savefig('result.png')

def main():
    MODEL_PATH = "./cuneiform_model_latest.pth"
    METADATA_PATH = "./images/cuneiform.json"
    TRUTH_TABLE_PATH = "./cuneiform_desc.txt"
    
    IMAGE_PATH = random.choice(
        [os.path.join("./images/", f) for f in os.listdir("./images/") if f.endswith(".png") and f.startswith("cuneiform_")]
    )
    
    logger.info(f"Specimen: {IMAGE_PATH}")
    predictor = CuneiformPredictor(MODEL_PATH, METADATA_PATH, TRUTH_TABLE_PATH)
    predictions = predictor.predict(IMAGE_PATH)
    
    print("\nTop Predictions:")
    for i, (unicode_char, confidence, description) in enumerate(predictions, 1):
        print(f"{i}. Unicode: {unicode_char}")
        print(f"   Confidence: {confidence:.2%}")
        print(f"   Description: {description}\n")
    
    predictor.visualize_results(IMAGE_PATH, predictions)

if __name__ == "__main__":
    main()
