import os
import requests
import torch
import torch.nn as nn
import torch.nn.functional as F
import open_clip
import numpy as np
import base64
from PIL import Image, ImageDraw, ImageFont
from io import BytesIO
import logging
import time
from contextlib import asynccontextmanager
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global variables
bioclip_model = None
classifier = None

# Configuration
CLASS_NAMES = [
    'Leaf-folder',
    'Pink-Bollworm', 
    'leaf folder - adult',
    'stem borer - adult',
    'stemborer'
]

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
GITHUB_MODEL_URL = "https://github.com/Vignesh-Vicky2004/Insect-Identification/raw/main/best_bioclip_classifier.pth"
LOCAL_MODEL_PATH = "best_bioclip_classifier.pth"

# Model Architecture
class OptimizedClassifier(nn.Module):
    def __init__(self, input_dim, num_classes, dropout_rate=0.4):
        super(OptimizedClassifier, self).__init__()
        self.classifier = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(input_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate * 0.5),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate * 0.3),
            nn.Linear(256, num_classes)
        )
    
    def forward(self, x):
        return self.classifier(x)

# Model Loading Functions
async def load_bioclip_models():
    global bioclip_model, classifier
    try:
        logger.info("Loading BioCLIP model...")
        bioclip_model, _, _ = open_clip.create_model_and_transforms(
            'hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224'
        )
        bioclip_model = bioclip_model.to(DEVICE)
        bioclip_model.eval()
        
        dummy_input = torch.randn(1, 3, 224, 224).to(DEVICE)
        with torch.no_grad():
            features = bioclip_model.encode_image(dummy_input)
            feature_dim = features.shape[-1]
        logger.info(f"BioCLIP loaded - Feature dimension: {feature_dim}")
        
        if not os.path.exists(LOCAL_MODEL_PATH):
            await download_trained_classifier()
        
        classifier = OptimizedClassifier(feature_dim, len(CLASS_NAMES))
        state_dict = torch.load(LOCAL_MODEL_PATH, map_location=DEVICE)
        classifier.load_state_dict(state_dict)
        classifier = classifier.to(DEVICE)
        classifier.eval()
        logger.info("Trained classifier loaded successfully")
        
        with torch.no_grad():
            test_features = bioclip_model.encode_image(dummy_input)
            test_output = classifier(test_features)
            test_probs = F.softmax(test_output, dim=1)
        logger.info(f"Pipeline test successful - Output shape: {test_probs.shape}")
        
    except Exception as e:
        logger.error(f"BioCLIP model loading failed: {e}")
        raise

async def download_trained_classifier():
    try:
        logger.info("Downloading trained classifier...")
        headers = {'User-Agent': 'Mozilla/5.0'}
        response = requests.get(GITHUB_MODEL_URL, headers=headers, stream=True, timeout=300)
        response.raise_for_status()
        
        with open(LOCAL_MODEL_PATH, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
        logger.info(f"Classifier downloaded successfully")
    except Exception as e:
        logger.error(f"Download failed: {e}")
        raise

# Image Processing Functions
def get_bioclip_transforms():
    import torchvision.transforms as transforms
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

def create_prediction_visualization(image: Image.Image, prediction: dict) -> str:
    try:
        annotated = image.copy()
        draw = ImageDraw.Draw(annotated)
        try:
            font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 20)
        except:
            font = ImageFont.load_default()
        
        species = prediction['species']
        confidence = prediction['confidence_percentage']
        main_text = f"Species: {species}"
        conf_text = f"Confidence: {confidence:.1f}%"
        
        if confidence >= 90:
            bg_color, text_color = "#27AE60", "white"
        elif confidence >= 70:
            bg_color, text_color = "#F39C12", "white"
        else:
            bg_color, text_color = "#E74C3C", "white"
        
        x, y = 10, 10
        padding = 10
        main_bbox = draw.textbbox((0, 0), main_text, font=font)
        conf_bbox = draw.textbbox((0, 0), conf_text, font=font)
        width = max(main_bbox[2] - main_bbox[0], conf_bbox[2] - conf_bbox[0]) + 2 * padding
        height = (main_bbox[3] - main_bbox[1]) + (conf_bbox[3] - conf_bbox[1]) + 3 * padding
        
        draw.rectangle([x, y, x + width, y + height], fill=bg_color)
        draw.text((x + padding, y + padding), main_text, fill=text_color, font=font)
        draw.text((x + padding, y + padding + main_bbox[3] - main_bbox[1] + padding), 
                 conf_text, fill=text_color, font=font)
        
        buffer = BytesIO()
        annotated.save(buffer, format='JPEG', quality=95)
        return base64.b64encode(buffer.getvalue()).decode()
    except Exception as e:
        logger.error(f"Visualization failed: {e}")
        buffer = BytesIO()
        image.save(buffer, format='JPEG', quality=95)
        return base64.b64encode(buffer.getvalue()).decode()

# FastAPI Application
@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        logger.info("Starting BioCLIP API...")
        await load_bioclip_models()
        logger.info("BioCLIP API ready!")
        yield
    except Exception as e:
        logger.error(f"Startup failed: {e}")
        yield
    finally:
        logger.info("Shutting down...")

app = FastAPI(
    title="BioCLIP Pest Identification API",
    version="3.0.0",
    description="API for pest identification using BioCLIP",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# API Endpoints
@app.get("/")
async def root():
    return {
        "message": "BioCLIP Pest Identification API",
        "version": "3.0.0",
        "status": "running",
        "model_info": {
            "foundation_model": "BioCLIP",
            "classes": CLASS_NAMES,
            "device": str(DEVICE)
        }
    }

@app.get("/health")
async def health_check():
    global bioclip_model, classifier
    if bioclip_model is None or classifier is None:
        return JSONResponse(
            status_code=503,
            content={"healthy": False, "status": "BioCLIP models not loaded"}
        )
    try:
        test_input = torch.randn(1, 3, 224, 224).to(DEVICE)
        with torch.no_grad():
            features = bioclip_model.encode_image(test_input)
            outputs = classifier(features)
        return {"healthy": True, "status": "ready", "device": str(DEVICE)}
    except Exception as e:
        return JSONResponse(
            status_code=503,
            content={"healthy": False, "error": str(e)}
        )

@app.post("/predict")
async def predict_pest(image: UploadFile = File(...)):
    global bioclip_model, classifier
    if bioclip_model is None or classifier is None:
        raise HTTPException(status_code=503, detail="BioCLIP models not loaded")
    
    if not image.content_type.startswith('image/'):
        raise HTTPException(status_code=400, detail="File must be an image")
    
    try:
        image_bytes = await image.read()
        pil_image = Image.open(BytesIO(image_bytes)).convert('RGB')
        transform = get_bioclip_transforms()
        image_tensor = transform(pil_image).unsqueeze(0).to(DEVICE)
        
        start_time = time.time()
        with torch.no_grad():
            features = bioclip_model.encode_image(image_tensor)
            outputs = classifier(features)
            probabilities = F.softmax(outputs, dim=1)
            confidence, predicted_class = torch.max(probabilities, 1)
            predicted_idx = predicted_class.item()
            predicted_species = CLASS_NAMES[predicted_idx]
        
        processing_time = time.time() - start_time
        prediction_result = {
            "species": predicted_species,
            "confidence": round(confidence.item(), 4),
            "confidence_percentage": round(confidence.item() * 100, 2)
        }
        
        annotated_image = create_prediction_visualization(pil_image, prediction_result)
        
        return {
            "success": True,
            "prediction": prediction_result,
            "annotated_image": annotated_image,
            "processing_time": round(processing_time, 3),
            "device": str(DEVICE)
        }
    except Exception as e:
        logger.error(f"Prediction failed: {e}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
