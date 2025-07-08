import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from fastapi import UploadFile, HTTPException
from fastapi.responses import JSONResponse
from typing import Dict, Any, List
from PIL import Image, ImageDraw, ImageFont
import base64
from io import BytesIO
import open_clip
import albumentations as A
from albumentations.pytorch import ToTensorV2
import time
import os
import logging

logger = logging.getLogger(__name__)

class ModelConfig:
    BIOCLIP_MODEL = 'hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224'
    MODEL_PATH = 'best_bioclip_classifier.pth'
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    CLASS_NAMES = [
        'Leaf-folder',
        'Pink-Bollworm',
        'leaf folder - adult',
        'stem borer - adult',
        'stemborer'
    ]
    CONFIDENCE_THRESHOLD = 0.5


class OptimizedClassifier(nn.Module):
    def __init__(self, input_dim, num_classes):
        super().__init__()
        self.model = nn.Sequential(
            nn.Dropout(0.4),
            nn.Linear(input_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.12),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        return self.model(x)


class BioCLIPPestService:
    def __init__(self):
        self.device = ModelConfig.DEVICE
        self.class_names = ModelConfig.CLASS_NAMES
        self._load_models()
        self._setup_transforms()

    def _load_models(self):
        self.clip_model, _, _ = open_clip.create_model_and_transforms(ModelConfig.BIOCLIP_MODEL)
        self.clip_model.to(self.device).eval()

        dummy = torch.randn(1, 3, 224, 224).to(self.device)
        self.feature_dim = self.clip_model.encode_image(dummy).shape[-1]

        self.classifier = OptimizedClassifier(self.feature_dim, len(self.class_names)).to(self.device)
        state = torch.load(ModelConfig.MODEL_PATH, map_location=self.device)
        self.classifier.load_state_dict(state)
        self.classifier.eval()

    def _setup_transforms(self):
        self.transforms = A.Compose([
            A.Resize(224, 224),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2()
        ])

    def _preprocess(self, image: Image.Image) -> torch.Tensor:
        np_img = np.array(image.convert("RGB"))
        tensor = self.transforms(image=np_img)["image"].unsqueeze(0).to(self.device)
        return tensor

    def _annotate(self, image: Image.Image, label: str, confidence: float) -> str:
        draw = ImageDraw.Draw(image)
        font = ImageFont.load_default()
        label_text = f"{label} ({confidence * 100:.1f}%)"
        draw.text((10, 10), label_text, fill="white", font=font)
        buffer = BytesIO()
        image.save(buffer, format="JPEG")
        return base64.b64encode(buffer.getvalue()).decode()

    def _predict(self, tensor: torch.Tensor) -> Dict[str, Any]:
        with torch.no_grad():
            features = self.clip_model.encode_image(tensor)
            logits = self.classifier(features)
            probs = F.softmax(logits, dim=1)
            conf, pred = torch.max(probs, 1)

        return {
            "label": self.class_names[pred.item()],
            "confidence": conf.item(),
            "all_probs": probs.squeeze().cpu().numpy()
        }

    async def identify_single(self, file: UploadFile) -> Dict[str, Any]:
        image = Image.open(BytesIO(await file.read()))
        tensor = self._preprocess(image)
        start = time.time()
        result = self._predict(tensor)
        elapsed = round(time.time() - start, 3)

        annotated = self._annotate(image, result["label"], result["confidence"])
        return {
            "species": result["label"],
            "confidence": round(result["confidence"], 4),
            "is_confident": result["confidence"] >= ModelConfig.CONFIDENCE_THRESHOLD,
            "confidence_percentage": round(result["confidence"] * 100, 2),
            "all_predictions": [
                {"species": cls, "confidence": round(float(prob), 4), "confidence_percentage": round(float(prob) * 100, 2)}
                for cls, prob in zip(self.class_names, result["all_probs"])
            ],
            "annotated_image": annotated,
            "processing_time": elapsed,
            "device": str(self.device)
        }

    async def identify_batch(self, files: List[UploadFile]) -> List[Dict[str, Any]]:
        return [await self.identify_single(file) for file in files]

    def get_model_info(self):
        return {
            "model_type": "BioCLIP + Optimized Classifier",
            "foundation_model": ModelConfig.BIOCLIP_MODEL,
            "classifier_path": ModelConfig.MODEL_PATH,
            "classes": self.class_names,
            "confidence_threshold": ModelConfig.CONFIDENCE_THRESHOLD,
            "device": str(self.device),
            "feature_dimension": self.feature_dim,
            "input_size": "224x224"
        }
