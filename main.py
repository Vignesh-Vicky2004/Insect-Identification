import os
import requests
from contextlib import asynccontextmanager
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import torch
import torch.nn as nn
import torch.nn.functional as F
import open_clip
import albumentations as A
from albumentations.pytorch import ToTensorV2
import numpy as np
import base64
from PIL import Image, ImageDraw, ImageFont
from io import BytesIO
import logging
import time
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global variables for model loading
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

# GitHub model URL - Update with your actual GitHub raw file URL
GITHUB_MODEL_URL = "https://github.com/Vignesh-Vicky2004/Insect-Identification/raw/main/best_bioclip_classifier.pth"
LOCAL_MODEL_PATH = "best_bioclip_classifier.pth"
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Define the OptimizedClassifier class (from your training code)
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

async def download_model_from_github():
    """Download model file from GitHub if not exists locally"""
    if os.path.exists(LOCAL_MODEL_PATH):
        logger.info(f"Model file already exists: {LOCAL_MODEL_PATH}")
        return LOCAL_MODEL_PATH
    
    try:
        logger.info(f"Downloading model from GitHub: {GITHUB_MODEL_URL}")
        response = requests.get(GITHUB_MODEL_URL, stream=True)
        response.raise_for_status()
        
        # Save model file
        with open(LOCAL_MODEL_PATH, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        
        logger.info(f"Model downloaded successfully: {LOCAL_MODEL_PATH}")
        return LOCAL_MODEL_PATH
        
    except Exception as e:
        logger.error(f"Failed to download model from GitHub: {e}")
        raise HTTPException(status_code=500, detail=f"Model download failed: {e}")

async def load_models():
    """Load models during startup"""
    global bioclip_model, classifier
    
    try:
        # Download model from GitHub if needed
        model_path = await download_model_from_github()
        
        # Load BioCLIP foundation model
        logger.info("Loading BioCLIP foundation model...")
        bioclip_model, _, _ = open_clip.create_model_and_transforms(
            'hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224'
        )
        bioclip_model = bioclip_model.to(DEVICE)
        bioclip_model.eval()
        
        # Get feature dimension
        dummy_input = torch.randn(1, 3, 224, 224).to(DEVICE)
        with torch.no_grad():
            features = bioclip_model.encode_image(dummy_input)
            feature_dim = features.shape[-1]
        
        # Load custom classifier
        logger.info(f"Loading classifier from {model_path}...")
        classifier = OptimizedClassifier(feature_dim, len(CLASS_NAMES)).to(DEVICE)
        classifier.load_state_dict(torch.load(model_path, map_location=DEVICE))
        classifier.eval()
        
        logger.info("Models loaded successfully")
        
    except Exception as e:
        logger.error(f"Failed to load models: {e}")
        raise

def setup_image_transforms():
    """Setup image preprocessing transforms"""
    return A.Compose([
        A.Resize(224, 224),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ])

@asynccontextmanager
async def lifespan(app: FastAPI):
    """TrueFoundry lifespan event handler"""
    await load_models()
    yield

# Create FastAPI app
app = FastAPI(
    title="BioCLIP Pest Identification API",
    version="2.0.0",
    description="AI-powered pest identification using BioCLIP foundation model",
    lifespan=lifespan,
    root_path=os.getenv("TFY_SERVICE_ROOT_PATH", "")
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "BioCLIP Pest Identification API",
        "version": "2.0.0",
        "status": "running",
        "classes": CLASS_NAMES
    }

@app.get("/health")
async def health():
    """Health check endpoint for TrueFoundry readiness/liveness probes"""
    global bioclip_model, classifier
    
    if bioclip_model is None or classifier is None:
        return JSONResponse(
            status_code=503,
            content={"healthy": False, "status": "models not loaded"}
        )
    
    return {"healthy": True, "status": "ready", "models_loaded": True}

@app.post("/predict")
async def predict_pest(image: UploadFile = File(...)):
    """Main prediction endpoint"""
    global bioclip_model, classifier
    
    # Check if models are loaded
    if bioclip_model is None or classifier is None:
        raise HTTPException(status_code=503, detail="Models not loaded")
    
    # Validate image
    if not image.content_type or not image.content_type.startswith('image/'):
        raise HTTPException(status_code=400, detail="File must be an image")
    
    try:
        # Read and process image
        image_bytes = await image.read()
        pil_image = Image.open(BytesIO(image_bytes)).convert('RGB')
        
        # Preprocess image
        transform = setup_image_transforms()
        image_np = np.array(pil_image)
        transformed = transform(image=image_np)
        image_tensor = transformed['image'].unsqueeze(0).to(DEVICE)
        
        # Run inference
        start_time = time.time()
        with torch.no_grad():
            # Extract features using BioCLIP
            features = bioclip_model.encode_image(image_tensor)
            
            # Classify using trained classifier
            outputs = classifier(features)
            probabilities = F.softmax(outputs, dim=1)
            
            # Get predictions
            confidence, predicted_class = torch.max(probabilities, 1)
            confidence_score = confidence.item()
            predicted_idx = predicted_class.item()
            predicted_class_name = CLASS_NAMES[predicted_idx]
            
            # Get all class probabilities
            all_probabilities = probabilities.squeeze().cpu().numpy()
        
        processing_time = time.time() - start_time
        
        # Format response
        response = {
            "success": True,
            "prediction": {
                "species": predicted_class_name,
                "confidence": round(confidence_score, 4),
                "confidence_percentage": round(confidence_score * 100, 2)
            },
            "all_predictions": [
                {
                    "species": class_name,
                    "confidence": round(float(prob), 4),
                    "confidence_percentage": round(float(prob) * 100, 2)
                }
                for class_name, prob in zip(CLASS_NAMES, all_probabilities)
            ],
            "processing_time": round(processing_time, 3),
            "image_info": {
                "width": pil_image.size[0],
                "height": pil_image.size[1]
            }
        }
        
        return JSONResponse(content=response)
        
    except Exception as e:
        logger.error(f"Prediction failed: {e}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
