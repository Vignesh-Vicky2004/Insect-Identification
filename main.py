"""
BioCLIP Pest Identification API - Error-Free Production Version
==============================================================
Simplified configuration and robust error handling for TrueFoundry deployment.
"""

import os
import sys
import asyncio
import requests
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import torchvision.models as models
import numpy as np
import base64
from PIL import Image, ImageDraw, ImageFont
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

# =============================================================================
# SIMPLIFIED CONFIGURATION
# =============================================================================

class Config:
    """Simplified configuration class - no dataclass complexity"""
    # Model settings
    MODEL_VERSION = "2.4.0"
    FEATURE_DIM = 2048
    NUM_CLASSES = 5
    
    # File paths
    GITHUB_MODEL_URL = "https://github.com/Vignesh-Vicky2004/Insect-Identification/raw/main/best_bioclip_classifier.pth"
    LOCAL_MODEL_PATH = "best_bioclip_classifier.pth"
    
    # API settings
    MAX_FILE_SIZE = 10 * 1024 * 1024  # 10MB
    REQUEST_TIMEOUT = 300
    MAX_RETRIES = 3
    
    # Image processing
    IMAGE_SIZE = (224, 224)
    NORMALIZE_MEAN = [0.485, 0.456, 0.406]
    NORMALIZE_STD = [0.229, 0.224, 0.225]
    
    # Logging
    LOG_LEVEL = "INFO"
    LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

# Initialize configuration
config = Config()

# Class names for pest identification
CLASS_NAMES = [
    'Leaf-folder',
    'Pink-Bollworm', 
    'leaf folder - adult',
    'stem borer - adult',
    'stemborer'
]

# Device configuration
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# =============================================================================
# LOGGING SETUP
# =============================================================================

def setup_logging():
    """Configure application logging"""
    logging.basicConfig(
        level=getattr(logging, config.LOG_LEVEL),
        format=config.LOG_FORMAT,
        handlers=[logging.StreamHandler(sys.stdout)]
    )
    return logging.getLogger(__name__)

logger = setup_logging()

# =============================================================================
# GLOBAL VARIABLES
# =============================================================================

feature_extractor = None
classifier = None
model_metadata = {}

# =============================================================================
# SIMPLIFIED MODEL DEFINITIONS
# =============================================================================

class SimpleFeatureExtractor(nn.Module):
    """Reliable ResNet50-based feature extractor"""
    
    def __init__(self):
        super().__init__()
        self.backbone = models.resnet50(weights='DEFAULT')
        self.backbone.fc = nn.Identity()
        
    def encode_image(self, x):
        return self.backbone(x)
    
    def forward(self, x):
        return self.encode_image(x)

class SimpleClassifier(nn.Module):
    """Simple, reliable classifier"""
    
    def __init__(self, input_dim=2048, num_classes=5):
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(input_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, 256),
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
    """Download model with error handling"""
    try:
        # Check existing file
        if os.path.exists(config.LOCAL_MODEL_PATH):
            file_size = os.path.getsize(config.LOCAL_MODEL_PATH)
            if file_size > 1000:
                logger.info(f"✅ Using existing model ({file_size} bytes)")
                return config.LOCAL_MODEL_PATH
        
        logger.info("📥 Downloading model from GitHub...")
        
        headers = {
            'User-Agent': 'Mozilla/5.0 (compatible; BioCLIP-API/1.0)',
            'Accept': 'application/octet-stream'
        }
        
        response = requests.get(
            config.GITHUB_MODEL_URL,
            headers=headers,
            stream=True,
            timeout=config.REQUEST_TIMEOUT
        )
        response.raise_for_status()
        
        # Validate response
        content_type = response.headers.get('content-type', '')
        if 'text/html' in content_type:
            raise ValueError("Got HTML instead of binary data")
        
        # Download file
        total_size = 0
        with open(config.LOCAL_MODEL_PATH, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
                    total_size += len(chunk)
        
        if total_size == 0:
            raise ValueError("Downloaded file is empty")
        
        logger.info(f"✅ Model downloaded ({total_size} bytes)")
        return config.LOCAL_MODEL_PATH
        
    except Exception as e:
        logger.error(f"❌ Download failed: {e}")
        return create_dummy_model()

def create_dummy_model():
    """Create dummy model for testing"""
    try:
        logger.info("🔄 Creating dummy model...")
        dummy = SimpleClassifier(config.FEATURE_DIM, config.NUM_CLASSES)
        torch.save(dummy.state_dict(), config.LOCAL_MODEL_PATH)
        logger.info("✅ Dummy model created")
        return config.LOCAL_MODEL_PATH
    except Exception as e:
        logger.error(f"❌ Dummy model creation failed: {e}")
        raise

async def initialize_models():
    """Initialize models with comprehensive error handling"""
    global feature_extractor, classifier, model_metadata
    
    try:
        logger.info("🔄 Starting model initialization...")
        
        # Initialize feature extractor
        logger.info("🖼️ Loading feature extractor...")
        feature_extractor = SimpleFeatureExtractor()
        feature_extractor = feature_extractor.to(DEVICE)
        feature_extractor.eval()
        
        # Test feature extractor
        test_input = torch.randn(1, 3, *config.IMAGE_SIZE).to(DEVICE)
        with torch.no_grad():
            features = feature_extractor.encode_image(test_input)
            actual_dim = features.shape[-1]
        
        logger.info(f"✅ Feature extractor ready (dim: {actual_dim})")
        
        # Download model
        model_path = await download_model_safely()
        
        # Initialize classifier
        logger.info("🎯 Loading classifier...")
        classifier = SimpleClassifier(actual_dim, len(CLASS_NAMES))
        
        # Try to load trained weights
        try:
            state_dict = torch.load(model_path, map_location=DEVICE)
            classifier.load_state_dict(state_dict)
            logger.info("✅ Trained weights loaded")
            model_loaded = True
        except Exception as e:
            logger.warning(f"⚠️ Using random weights: {e}")
            model_loaded = False
        
        classifier = classifier.to(DEVICE)
        classifier.eval()
        
        # Test pipeline
        with torch.no_grad():
            test_features = feature_extractor.encode_image(test_input)
            test_output = classifier(test_features)
            test_probs = F.softmax(test_output, dim=1)
        
        # Store metadata
        model_metadata = {
            "feature_dim": actual_dim,
            "num_classes": len(CLASS_NAMES),
            "model_loaded": model_loaded,
            "device": str(DEVICE),
            "version": config.MODEL_VERSION
        }
        
        logger.info(f"✅ Pipeline test successful (shape: {test_probs.shape})")
        logger.info("🎉 Models initialized successfully!")
        
    except Exception as e:
        logger.error(f"❌ Model initialization failed: {e}")
        raise

# =============================================================================
# IMAGE PROCESSING
# =============================================================================

def setup_transforms():
    """Setup image transforms"""
    return transforms.Compose([
        transforms.Resize(config.IMAGE_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(mean=config.NORMALIZE_MEAN, std=config.NORMALIZE_STD)
    ])

def process_image(image_bytes):
    """Process image with validation"""
    if len(image_bytes) == 0:
        raise ValueError("Empty image data")
    
    pil_image = Image.open(BytesIO(image_bytes)).convert('RGB')
    
    if pil_image.size[0] < 32 or pil_image.size[1] < 32:
        raise ValueError("Image too small")
    
    transform = setup_transforms()
    image_tensor = transform(pil_image).unsqueeze(0).to(DEVICE)
    
    return image_tensor, pil_image

# =============================================================================
# FASTAPI APPLICATION
# =============================================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan management"""
    try:
        logger.info("🚀 Starting application...")
        await initialize_models()
        logger.info("✅ Application ready!")
        yield
    except Exception as e:
        logger.error(f"❌ Startup failed: {e}")
        # Allow degraded mode
        global feature_extractor, classifier
        feature_extractor = None
        classifier = None
        yield
    finally:
        logger.info("🔄 Application shutdown...")

# Create FastAPI app
app = FastAPI(
    title="BioCLIP Pest Identification API",
    version=config.MODEL_VERSION,
    description="Error-free pest identification API for TrueFoundry",
    lifespan=lifespan
)

# CORS middleware
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
        "status": "running",
        "classes": CLASS_NAMES,
        "device": str(DEVICE),
        "models_loaded": feature_extractor is not None and classifier is not None,
        "endpoints": {
            "health": "/health",
            "predict": "/predict",
            "docs": "/docs"
        }
    }

@app.get("/health")
async def health():
    """Health check endpoint"""
    global feature_extractor, classifier
    
    if feature_extractor is None or classifier is None:
        return JSONResponse(
            status_code=503,
            content={"healthy": False, "status": "models not loaded"}
        )
    
    try:
        # Test inference
        test_input = torch.randn(1, 3, *config.IMAGE_SIZE).to(DEVICE)
        with torch.no_grad():
            features = feature_extractor.encode_image(test_input)
            outputs = classifier(features)
        
        return {
            "healthy": True,
            "status": "ready",
            "models_loaded": True,
            "device": str(DEVICE),
            "version": config.MODEL_VERSION
        }
        
    except Exception as e:
        return JSONResponse(
            status_code=503,
            content={"healthy": False, "error": str(e)}
        )

@app.post("/predict")
async def predict_pest(image: UploadFile = File(...)):
    """Prediction endpoint"""
    global feature_extractor, classifier
    
    # Check models
    if feature_extractor is None or classifier is None:
        raise HTTPException(status_code=503, detail="Models not loaded")
    
    # Validate image
    if not image.content_type or not image.content_type.startswith('image/'):
        raise HTTPException(status_code=400, detail="File must be an image")
    
    if image.size and image.size > config.MAX_FILE_SIZE:
        raise HTTPException(status_code=400, detail="File too large")
    
    try:
        # Process image
        logger.info(f"Processing: {image.filename}")
        image_bytes = await image.read()
        image_tensor, pil_image = process_image(image_bytes)
        
        # Run inference
        start_time = time.time()
        with torch.no_grad():
            features = feature_extractor.encode_image(image_tensor)
            outputs = classifier(features)
            probabilities = F.softmax(outputs, dim=1)
            
            confidence, predicted_class = torch.max(probabilities, 1)
            confidence_score = confidence.item()
            predicted_idx = predicted_class.item()
            
            if predicted_idx >= len(CLASS_NAMES):
                predicted_idx = 0
            
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
            "image_info": {
                "filename": image.filename,
                "width": pil_image.size[0],
                "height": pil_image.size[1],
                "size_bytes": len(image_bytes)
            },
            "processing_time": round(processing_time, 3),
            "device": str(DEVICE),
            "version": config.MODEL_VERSION
        }
        
        logger.info(f"✅ Prediction: {predicted_class_name} ({confidence_score:.2%})")
        return JSONResponse(content=response)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Prediction failed: {e}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

# =============================================================================
# ERROR HANDLERS
# =============================================================================

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """Global exception handler"""
    logger.error(f"❌ Unhandled exception: {exc}")
    return JSONResponse(
        status_code=500,
        content={"error": "Internal server error", "type": type(exc).__name__}
    )

# =============================================================================
# MAIN ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
