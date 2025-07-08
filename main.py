import os
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

# Global variables for model loading
bioclip_model = None
classifier = None

# TrueFoundry environment variables
MODEL_PATH = os.getenv("MODEL_PATH", "best_bioclip_classifier.pth")
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

async def load_models():
    """Load models during startup"""
    global bioclip_model, classifier
    
    try:
        # Load BioCLIP foundation model
        bioclip_model, _, _ = open_clip.create_model_and_transforms(
            'hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224'
        )
        bioclip_model = bioclip_model.to(DEVICE)
        bioclip_model.eval()
        
        # Load classifier
        dummy_input = torch.randn(1, 3, 224, 224).to(DEVICE)
        with torch.no_grad():
            features = bioclip_model.encode_image(dummy_input)
            feature_dim = features.shape[-1]
        
        classifier = OptimizedClassifier(feature_dim, 5).to(DEVICE)
        classifier.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
        classifier.eval()
        
        logging.info("Models loaded successfully")
        
    except Exception as e:
        logging.error(f"Failed to load models: {e}")
        raise

@asynccontextmanager
async def lifespan(app: FastAPI):
    """TrueFoundry lifespan event handler"""
    await load_models()
    yield

# Create FastAPI app with TrueFoundry configuration
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

# Health check endpoint (required for TrueFoundry)
@app.get("/health")
async def health():
    """Health check endpoint for TrueFoundry readiness/liveness probes"""
    return {"healthy": True, "status": "ready"}

# Your existing routes...
@app.post("/predict")
async def predict_pest(image: UploadFile = File(...)):
    """Main prediction endpoint"""
    # Your existing prediction logic here
    pass

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
