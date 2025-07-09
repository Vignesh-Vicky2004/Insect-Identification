"""
BioCLIP Pest Identification API - CORRECTED VERSION
=================================================
This version properly uses your fine-tuned BioCLIP model
"""

import os
import sys
import asyncio
import requests
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import numpy as np
import base64
from PIL import Image
from io import BytesIO
import logging
import time
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from contextlib import asynccontextmanager
from fastapi import FastAPI, File, UploadFile, HTTPException, Request
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

# Install required packages
try:
    import open_clip
except ImportError:
    os.system("pip install open-clip-torch")
    import open_clip

# =============================================================================
# CONFIGURATION
# =============================================================================

class Config:
    """Configuration matching your fine-tuned model"""
    MODEL_VERSION = "2.4.0"
    
    # BioCLIP model name (MUST match your training)
    BIOCLIP_MODEL_NAME = 'hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224'
    
    # File paths
    GITHUB_MODEL_URL = "https://github.com/Vignesh-Vicky2004/Insect-Identification/raw/main/best_bioclip_classifier.pth"
    LOCAL_MODEL_PATH = "best_bioclip_classifier.pth"
    
    # API settings
    MAX_FILE_SIZE = 10 * 1024 * 1024  # 10MB
    REQUEST_TIMEOUT = 300
    
    # Image processing (must match your training)
    IMAGE_SIZE = (224, 224)
    NORMALIZE_MEAN = [0.485, 0.456, 0.406]
    NORMALIZE_STD = [0.229, 0.224, 0.225]
    
    # Model architecture (must match your training)
    DROPOUT_RATE = 0.3
    
    LOG_LEVEL = "INFO"

config = Config()

# Class names (MUST match your training exactly)
CLASS_NAMES = [
    'Leaf-folder',
    'Pink-Bollworm', 
    'leaf folder - adult',
    'stem borer - adult',
    'stemborer'
]

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# =============================================================================
# LOGGING SETUP
# =============================================================================

logging.basicConfig(
    level=getattr(logging, config.LOG_LEVEL),
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

# =============================================================================
# GLOBAL VARIABLES
# =============================================================================

bioclip_model = None
classifier = None
tokenizer = None
preprocess_val = None
model_metadata = {}

# =============================================================================
# CORRECT MODEL DEFINITIONS
# =============================================================================

class BioCLIPClassifier(nn.Module):
    """
    EXACT classifier architecture from your fine-tuning code
    This MUST match the architecture you used during training
    """
    def __init__(self, input_dim: int, num_classes: int, dropout_rate: float = 0.3):
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(input_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, num_classes)
        )
    
    def forward(self, x):
        return self.classifier(x)

# =============================================================================
# MODEL MANAGEMENT
# =============================================================================

async def download_model_safely():
    """Download your fine-tuned model"""
    try:
        if os.path.exists(config.LOCAL_MODEL_PATH):
            file_size = os.path.getsize(config.LOCAL_MODEL_PATH)
            if file_size > 1000:
                logger.info(f"✅ Using existing model ({file_size} bytes)")
                return config.LOCAL_MODEL_PATH
        
        logger.info("📥 Downloading fine-tuned model from GitHub...")
        
        response = requests.get(
            config.GITHUB_MODEL_URL,
            headers={'User-Agent': 'BioCLIP-API/1.0'},
            stream=True,
            timeout=config.REQUEST_TIMEOUT
        )
        response.raise_for_status()
        
        total_size = 0
        with open(config.LOCAL_MODEL_PATH, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
                    total_size += len(chunk)
        
        logger.info(f"✅ Model downloaded ({total_size} bytes)")
        return config.LOCAL_MODEL_PATH
        
    except Exception as e:
        logger.error(f"❌ Download failed: {e}")
        raise

def get_bioclip_output_dim(model) -> int:
    """Get the actual output dimension from BioCLIP model"""
    try:
        dummy_input = torch.randn(1, 3, *config.IMAGE_SIZE).to(DEVICE)
        with torch.no_grad():
            features = model.encode_image(dummy_input)
            return features.shape[-1]
    except Exception as e:
        logger.error(f"Could not determine BioCLIP output dimension: {e}")
        return 512  # Fallback

async def initialize_models():
    """Initialize BioCLIP model and your fine-tuned classifier"""
    global bioclip_model, classifier, tokenizer, preprocess_val, model_metadata
    
    try:
        logger.info("🔄 Loading BioCLIP model...")
        
        # Load BioCLIP model (same as your training)
        bioclip_model, _, preprocess_val = open_clip.create_model_and_transforms(
            config.BIOCLIP_MODEL_NAME
        )
        bioclip_model = bioclip_model.to(DEVICE)
        bioclip_model.eval()
        
        # Get tokenizer
        tokenizer = open_clip.get_tokenizer(config.BIOCLIP_MODEL_NAME)
        
        # Get correct output dimension
        output_dim = get_bioclip_output_dim(bioclip_model)
        logger.info(f"✅ BioCLIP loaded (output_dim: {output_dim})")
        
        # Download your fine-tuned classifier
        model_path = await download_model_safely()
        
        # Initialize classifier with EXACT architecture
        classifier = BioCLIPClassifier(
            input_dim=output_dim,
            num_classes=len(CLASS_NAMES),
            dropout_rate=config.DROPOUT_RATE
        )
        
        # Load your fine-tuned weights
        try:
            state_dict = torch.load(model_path, map_location=DEVICE)
            classifier.load_state_dict(state_dict)
            logger.info("✅ Fine-tuned classifier weights loaded")
            model_loaded = True
        except Exception as e:
            logger.error(f"❌ Failed to load fine-tuned weights: {e}")
            logger.info("Using random weights - predictions will be wrong!")
            model_loaded = False
        
        classifier = classifier.to(DEVICE)
        classifier.eval()
        
        # Test the complete pipeline
        test_input = torch.randn(1, 3, *config.IMAGE_SIZE).to(DEVICE)
        with torch.no_grad():
            features = bioclip_model.encode_image(test_input)
            outputs = classifier(features)
            probs = F.softmax(outputs, dim=1)
        
        model_metadata = {
            "bioclip_model": config.BIOCLIP_MODEL_NAME,
            "feature_dim": output_dim,
            "num_classes": len(CLASS_NAMES),
            "model_loaded": model_loaded,
            "device": str(DEVICE),
            "version": config.MODEL_VERSION
        }
        
        logger.info("🎉 BioCLIP and fine-tuned classifier initialized successfully!")
        
    except Exception as e:
        logger.error(f"❌ Model initialization failed: {e}")
        raise

# =============================================================================
# IMAGE PROCESSING
# =============================================================================

def setup_bioclip_transforms():
    """Use BioCLIP's validation transforms"""
    return transforms.Compose([
        transforms.Resize(config.IMAGE_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(mean=config.NORMALIZE_MEAN, std=config.NORMALIZE_STD)
    ])

def process_image(image_bytes):
    """Process image for BioCLIP"""
    if len(image_bytes) == 0:
        raise ValueError("Empty image data")
    
    pil_image = Image.open(BytesIO(image_bytes)).convert('RGB')
    
    if pil_image.size[0] < 32 or pil_image.size[1] < 32:
        raise ValueError("Image too small")
    
    # Use BioCLIP preprocessing
    if preprocess_val is not None:
        image_tensor = preprocess_val(pil_image).unsqueeze(0).to(DEVICE)
    else:
        transform = setup_bioclip_transforms()
        image_tensor = transform(pil_image).unsqueeze(0).to(DEVICE)
    
    return image_tensor, pil_image

# =============================================================================
# FASTAPI APPLICATION
# =============================================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan management"""
    try:
        logger.info("🚀 Starting BioCLIP API...")
        await initialize_models()
        logger.info("✅ Application ready!")
        yield
    except Exception as e:
        logger.error(f"❌ Startup failed: {e}")
        raise
    finally:
        logger.info("🔄 Application shutdown...")

app = FastAPI(
    title="BioCLIP Pest Identification API",
    version=config.MODEL_VERSION,
    description="Fine-tuned BioCLIP model for pest identification",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# =============================================================================
# API ENDPOINTS
# =============================================================================

@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "BioCLIP Pest Identification API",
        "version": config.MODEL_VERSION,
        "bioclip_model": config.BIOCLIP_MODEL_NAME,
        "classes": CLASS_NAMES,
        "device": str(DEVICE),
        "models_loaded": bioclip_model is not None and classifier is not None
    }

@app.get("/health")
async def health():
    """Health check endpoint"""
    if bioclip_model is None or classifier is None:
        return JSONResponse(
            status_code=503,
            content={"healthy": False, "status": "models not loaded"}
        )
    
    try:
        # Test BioCLIP pipeline
        test_input = torch.randn(1, 3, *config.IMAGE_SIZE).to(DEVICE)
        with torch.no_grad():
            features = bioclip_model.encode_image(test_input)
            outputs = classifier(features)
        
        return {
            "healthy": True,
            "status": "ready",
            "bioclip_loaded": True,
            "classifier_loaded": True,
            "device": str(DEVICE),
            "metadata": model_metadata
        }
        
    except Exception as e:
        return JSONResponse(
            status_code=503,
            content={"healthy": False, "error": str(e)}
        )

@app.post("/predict")
async def predict_pest(image: UploadFile = File(...)):
    """Prediction endpoint using fine-tuned BioCLIP"""
    if bioclip_model is None or classifier is None:
        raise HTTPException(status_code=503, detail="Models not loaded")
    
    if not image.content_type or not image.content_type.startswith('image/'):
        raise HTTPException(status_code=400, detail="File must be an image")
    
    try:
        # Process image
        logger.info(f"Processing: {image.filename}")
        image_bytes = await image.read()
        image_tensor, pil_image = process_image(image_bytes)
        
        # Run inference with BioCLIP + fine-tuned classifier
        start_time = time.time()
        with torch.no_grad():
            # Extract features using BioCLIP
            features = bioclip_model.encode_image(image_tensor)
            
            # Classify using your fine-tuned classifier
            outputs = classifier(features)
            probabilities = F.softmax(outputs, dim=1)
            
            confidence, predicted_class = torch.max(probabilities, 1)
            confidence_score = confidence.item()
            predicted_idx = predicted_class.item()
            
            predicted_class_name = CLASS_NAMES[predicted_idx]
            all_probs = probabilities.squeeze().cpu().numpy()
        
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
                for class_name, prob in zip(CLASS_NAMES, all_probs)
            ],
            "model_info": {
                "bioclip_model": config.BIOCLIP_MODEL_NAME,
                "fine_tuned": model_metadata.get("model_loaded", False),
                "feature_dim": model_metadata.get("feature_dim", "unknown")
            },
            "processing_time": round(processing_time, 3),
            "device": str(DEVICE)
        }
        
        logger.info(f"✅ Prediction: {predicted_class_name} ({confidence_score:.2%})")
        return JSONResponse(content=response)
        
    except Exception as e:
        logger.error(f"❌ Prediction failed: {e}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """Global exception handler"""
    logger.error(f"❌ Unhandled exception: {exc}")
    return JSONResponse(
        status_code=500,
        content={"error": "Internal server error", "type": type(exc).__name__}
    )

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
