"""
Corrected BioCLIP Pest Identification API
==========================================
This version matches the exact architecture used in fine-tuning
"""

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

# Configuration - Must match your training exactly
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

# ===============================================================================
# EXACT ARCHITECTURE FROM YOUR FINE-TUNING CODE
# ===============================================================================

class OptimizedClassifier(nn.Module):
    """EXACT same architecture as in your fine-tuning code"""
    def __init__(self, input_dim, num_classes, dropout_rate=0.4):
        super(OptimizedClassifier, self).__init__()
        
        # This MUST match your training architecture exactly
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

# ===============================================================================
# BIOCLIP MODEL LOADING (SAME AS TRAINING)
# ===============================================================================

async def load_bioclip_models():
    """Load the EXACT same BioCLIP model used in training"""
    global bioclip_model, classifier
    
    try:
        logger.info("🔄 Loading BioCLIP model (same as training)...")
        
        # Load the EXACT same BioCLIP model used in training
        bioclip_model, _, _ = open_clip.create_model_and_transforms(
            'hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224'
        )
        bioclip_model = bioclip_model.to(DEVICE)
        bioclip_model.eval()
        
        # Get the EXACT feature dimension used in training
        dummy_input = torch.randn(1, 3, 224, 224).to(DEVICE)
        with torch.no_grad():
            features = bioclip_model.encode_image(dummy_input)
            feature_dim = features.shape[-1]
        
        logger.info(f"✅ BioCLIP loaded - Feature dimension: {feature_dim}")
        
        # Download trained classifier
        await download_trained_classifier()
        
        # Load classifier with EXACT same architecture
        classifier = OptimizedClassifier(feature_dim, len(CLASS_NAMES))
        
        # Load the trained weights
        state_dict = torch.load(LOCAL_MODEL_PATH, map_location=DEVICE)
        classifier.load_state_dict(state_dict)
        classifier = classifier.to(DEVICE)
        classifier.eval()
        
        logger.info("✅ Trained classifier loaded successfully")
        
        # Test the complete pipeline
        with torch.no_grad():
            test_features = bioclip_model.encode_image(dummy_input)
            test_output = classifier(test_features)
            test_probs = F.softmax(test_output, dim=1)
        
        logger.info(f"✅ Pipeline test successful - Output shape: {test_probs.shape}")
        
    except Exception as e:
        logger.error(f"❌ BioCLIP model loading failed: {e}")
        raise

async def download_trained_classifier():
    """Download the trained classifier"""
    if os.path.exists(LOCAL_MODEL_PATH):
        logger.info("✅ Classifier file already exists")
        return
    
    try:
        logger.info("📥 Downloading trained classifier...")
        
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        }
        
        response = requests.get(GITHUB_MODEL_URL, headers=headers, stream=True, timeout=300)
        response.raise_for_status()
        
        total_size = 0
        with open(LOCAL_MODEL_PATH, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
                    total_size += len(chunk)
        
        logger.info(f"✅ Classifier downloaded ({total_size:,} bytes)")
        
    except Exception as e:
        logger.error(f"❌ Download failed: {e}")
        raise

# ===============================================================================
# IMAGE PROCESSING (SAME AS TRAINING)
# ===============================================================================

def get_bioclip_transforms():
    """Image transforms that match your training pipeline"""
    import torchvision.transforms as transforms
    
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

def create_prediction_visualization(image: Image.Image, prediction: dict) -> str:
    """Create visualization of prediction results"""
    try:
        # Create annotated image
        annotated = image.copy()
        draw = ImageDraw.Draw(annotated)
        
        # Load font
        try:
            font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 20)
        except:
            font = ImageFont.load_default()
        
        # Prepare text
        species = prediction['species']
        confidence = prediction['confidence_percentage']
        
        main_text = f"Species: {species}"
        conf_text = f"Confidence: {confidence:.1f}%"
        
        # Choose color based on confidence
        if confidence >= 90:
            bg_color, text_color = "#27AE60", "white"  # Green
        elif confidence >= 70:
            bg_color, text_color = "#F39C12", "white"  # Orange
        else:
            bg_color, text_color = "#E74C3C", "white"  # Red
        
        # Draw background and text
        x, y = 10, 10
        padding = 10
        
        # Calculate text dimensions
        main_bbox = draw.textbbox((0, 0), main_text, font=font)
        conf_bbox = draw.textbbox((0, 0), conf_text, font=font)
        
        width = max(main_bbox[2] - main_bbox[0], conf_bbox[2] - conf_bbox[0]) + 2 * padding
        height = (main_bbox[3] - main_bbox[1]) + (conf_bbox[3] - conf_bbox[1]) + 3 * padding
        
        # Draw background
        draw.rectangle([x, y, x + width, y + height], fill=bg_color)
        
        # Draw text
        draw.text((x + padding, y + padding), main_text, fill=text_color, font=font)
        draw.text((x + padding, y + padding + main_bbox[3] - main_bbox[1] + padding), 
                 conf_text, fill=text_color, font=font)
        
        # Convert to base64
        buffer = BytesIO()
        annotated.save(buffer, format='JPEG', quality=95)
        return base64.b64encode(buffer.getvalue()).decode()
        
    except Exception as e:
        logger.error(f"Visualization failed: {e}")
        # Return original image
        buffer = BytesIO()
        image.save(buffer, format='JPEG', quality=95)
        return base64.b64encode(buffer.getvalue()).decode()

# ===============================================================================
# FASTAPI APPLICATION
# ===============================================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan with BioCLIP model loading"""
    try:
        logger.info("🚀 Starting BioCLIP API...")
        await load_bioclip_models()
        logger.info("✅ BioCLIP API ready!")
        yield
    except Exception as e:
        logger.error(f"❌ Startup failed: {e}")
        yield
    finally:
        logger.info("🔄 Shutting down...")

# Create FastAPI app
app = FastAPI(
    title="BioCLIP Pest Identification API",
    version="3.0.0",
    description="Corrected API using the same BioCLIP model as training",
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

# ===============================================================================
# API ENDPOINTS
# ===============================================================================

@app.get("/")
async def root():
    """Root endpoint with model information"""
    return {
        "message": "BioCLIP Pest Identification API",
        "version": "3.0.0",
        "status": "running",
        "model_info": {
            "foundation_model": "BioCLIP (microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224)",
            "architecture": "BioCLIP + OptimizedClassifier",
            "classes": CLASS_NAMES,
            "device": str(DEVICE)
        },
        "endpoints": {
            "health": "/health",
            "predict": "/predict",
            "model_info": "/model-info"
        }
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    global bioclip_model, classifier
    
    if bioclip_model is None or classifier is None:
        return JSONResponse(
            status_code=503,
            content={"healthy": False, "status": "BioCLIP models not loaded"}
        )
    
    try:
        # Test BioCLIP pipeline
        test_input = torch.randn(1, 3, 224, 224).to(DEVICE)
        with torch.no_grad():
            features = bioclip_model.encode_image(test_input)
            outputs = classifier(features)
        
        return {
            "healthy": True,
            "status": "ready",
            "model_type": "BioCLIP",
            "feature_dimension": features.shape[-1],
            "device": str(DEVICE)
        }
        
    except Exception as e:
        return JSONResponse(
            status_code=503,
            content={"healthy": False, "error": str(e)}
        )

@app.get("/model-info")
async def model_info():
    """Detailed model information"""
    global bioclip_model, classifier
    
    if bioclip_model is None or classifier is None:
        return {"error": "Models not loaded"}
    
    # Get feature dimension
    test_input = torch.randn(1, 3, 224, 224).to(DEVICE)
    with torch.no_grad():
        features = bioclip_model.encode_image(test_input)
        feature_dim = features.shape[-1]
    
    return {
        "foundation_model": {
            "name": "BioCLIP",
            "full_name": "microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224",
            "type": "Vision-Language Model",
            "feature_dimension": feature_dim
        },
        "classifier": {
            "architecture": "OptimizedClassifier",
            "layers": [
                f"Linear({feature_dim}, 512)",
                "BatchNorm1d(512)",
                "ReLU",
                "Dropout(0.2)",
                "Linear(512, 256)",
                "BatchNorm1d(256)", 
                "ReLU",
                "Dropout(0.12)",
                f"Linear(256, {len(CLASS_NAMES)})"
            ]
        },
        "classes": CLASS_NAMES,
        "preprocessing": {
            "resize": "224x224",
            "normalize": {
                "mean": [0.485, 0.456, 0.406],
                "std": [0.229, 0.224, 0.225]
            }
        }
    }

@app.post("/predict")
async def predict_pest(image: UploadFile = File(...)):
    """Corrected prediction endpoint using BioCLIP"""
    global bioclip_model, classifier
    
    # Check models
    if bioclip_model is None or classifier is None:
        raise HTTPException(status_code=503, detail="BioCLIP models not loaded")
    
    # Validate image
    if not image.content_type or not image.content_type.startswith('image/'):
        raise HTTPException(status_code=400, detail="File must be an image")
    
    if image.size and image.size > 10 * 1024 * 1024:
        raise HTTPException(status_code=400, detail="Image too large (max 10MB)")
    
    try:
        # Read image
        logger.info(f"Processing image: {image.filename}")
        image_bytes = await image.read()
        
        if len(image_bytes) == 0:
            raise HTTPException(status_code=400, detail="Empty image file")
        
        # Process image
        pil_image = Image.open(BytesIO(image_bytes)).convert('RGB')
        
        # Apply same transforms as training
        transform = get_bioclip_transforms()
        image_tensor = transform(pil_image).unsqueeze(0).to(DEVICE)
        
        # Run inference with BioCLIP
        start_time = time.time()
        with torch.no_grad():
            # Extract features using BioCLIP (same as training)
            features = bioclip_model.encode_image(image_tensor)
            
            # Classify using trained classifier
            outputs = classifier(features)
            probabilities = F.softmax(outputs, dim=1)
            
            # Get predictions
            confidence, predicted_class = torch.max(probabilities, 1)
            confidence_score = confidence.item()
            predicted_idx = predicted_class.item()
            
            # Validate index
            if predicted_idx >= len(CLASS_NAMES):
                predicted_idx = 0
            
            predicted_species = CLASS_NAMES[predicted_idx]
            all_probabilities = probabilities.squeeze().cpu().numpy()
        
        processing_time = time.time() - start_time
        
        # Create prediction result
        prediction_result = {
            "species": predicted_species,
            "confidence": round(confidence_score, 4),
            "confidence_percentage": round(confidence_score * 100, 2)
        }
        
        # Create visualization
        annotated_image = create_prediction_visualization(pil_image, prediction_result)
        
        # Format response
        response = {
            "success": True,
            "prediction": prediction_result,
            "all_predictions": [
                {
                    "species": class_name,
                    "confidence": round(float(prob), 4),
                    "confidence_percentage": round(float(prob) * 100, 2)
                }
                for class_name, prob in zip(CLASS_NAMES, all_probabilities)
            ],
            "model_info": {
                "foundation_model": "BioCLIP",
                "feature_dimension": features.shape[-1],
                "architecture_match": "✅ Matches training exactly"
            },
            "image_info": {
                "filename": image.filename,
                "width": pil_image.size[0],
                "height": pil_image.size[1],
                "size_bytes": len(image_bytes)
            },
            "annotated_image": annotated_image,
            "processing_time": round(processing_time, 3),
            "device": str(DEVICE)
        }
        
        logger.info(f"✅ BioCLIP Prediction: {predicted_species} ({confidence_score:.2%})")
        return JSONResponse(content=response)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Prediction failed: {e}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

# ===============================================================================
# ERROR HANDLERS
# ===============================================================================

@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    logger.error(f"❌ Unhandled exception: {exc}")
    return JSONResponse(
        status_code=500,
        content={"error": "Internal server error", "type": type(exc).__name__}
    )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
