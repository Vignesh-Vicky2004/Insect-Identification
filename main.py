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
import torchvision.transforms as transforms
import numpy as np
import base64
from PIL import Image, ImageDraw, ImageFont
from io import BytesIO
import logging
import time
import traceback
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
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

# CORRECTED GitHub model URL - using /raw/ instead of /blob/
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
        logger.info(f"✅ Model file already exists: {LOCAL_MODEL_PATH}")
        return LOCAL_MODEL_PATH
    
    try:
        logger.info(f"📥 Downloading model from GitHub: {GITHUB_MODEL_URL}")
        response = requests.get(GITHUB_MODEL_URL, stream=True, timeout=300)
        response.raise_for_status()
        
        # Check if we got HTML instead of binary data (wrong URL format)
        content_type = response.headers.get('content-type', '')
        if 'text/html' in content_type:
            raise ValueError("Got HTML response instead of binary file - check GitHub URL format")
        
        # Save model file
        file_size = 0
        with open(LOCAL_MODEL_PATH, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
                    file_size += len(chunk)
        
        logger.info(f"✅ Model downloaded successfully: {LOCAL_MODEL_PATH} ({file_size} bytes)")
        return LOCAL_MODEL_PATH
        
    except Exception as e:
        logger.error(f"❌ Failed to download model from GitHub: {e}")
        raise HTTPException(status_code=500, detail=f"Model download failed: {e}")

async def load_models():
    """Load models during startup with comprehensive logging"""
    global bioclip_model, classifier
    
    try:
        logger.info("🔄 Starting model loading process...")
        logger.info(f"🖥️ Device: {DEVICE}")
        logger.info(f"🔢 CUDA available: {torch.cuda.is_available()}")
        
        # Download model from GitHub if needed
        logger.info("📥 Checking model file...")
        model_path = await download_model_from_github()
        
        # Verify model file exists and is not empty
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")
        
        file_size = os.path.getsize(model_path)
        if file_size == 0:
            raise ValueError(f"Model file is empty: {model_path}")
        
        logger.info(f"✅ Model file verified: {model_path} ({file_size} bytes)")
        
        # Load BioCLIP foundation model
        logger.info("🤖 Loading BioCLIP foundation model...")
        try:
            bioclip_model, _, _ = open_clip.create_model_and_transforms(
                'hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224'
            )
            logger.info("🚀 BioCLIP model loaded, moving to device...")
            bioclip_model = bioclip_model.to(DEVICE)
            bioclip_model.eval()
            logger.info(f"✅ BioCLIP model ready on {DEVICE}")
        except Exception as e:
            logger.error(f"❌ Failed to load BioCLIP: {e}")
            raise
        
        # Get feature dimension
        logger.info("🔍 Getting feature dimensions...")
        try:
            dummy_input = torch.randn(1, 3, 224, 224).to(DEVICE)
            with torch.no_grad():
                features = bioclip_model.encode_image(dummy_input)
                feature_dim = features.shape[-1]
            logger.info(f"📏 Feature dimension: {feature_dim}")
        except Exception as e:
            logger.error(f"❌ Failed to get feature dimensions: {e}")
            raise
        
        # Load custom classifier
        logger.info(f"🎯 Loading classifier from {model_path}...")
        try:
            classifier = OptimizedClassifier(feature_dim, len(CLASS_NAMES)).to(DEVICE)
            state_dict = torch.load(model_path, map_location=DEVICE)
            classifier.load_state_dict(state_dict)
            classifier.eval()
            logger.info("✅ Classifier loaded successfully")
        except Exception as e:
            logger.error(f"❌ Failed to load classifier: {e}")
            raise
        
        # Test inference to ensure everything works
        logger.info("🧪 Testing model inference...")
        try:
            with torch.no_grad():
                test_features = bioclip_model.encode_image(dummy_input)
                test_output = classifier(test_features)
                test_probs = F.softmax(test_output, dim=1)
                logger.info(f"✅ Test inference successful, output shape: {test_probs.shape}")
        except Exception as e:
            logger.error(f"❌ Test inference failed: {e}")
            raise
        
        logger.info("🎉 All models loaded and tested successfully!")
        
    except Exception as e:
        logger.error(f"❌ Failed to load models: {e}")
        logger.error(f"📍 Traceback: {traceback.format_exc()}")
        raise

def setup_image_transforms():
    """Setup image preprocessing transforms without OpenCV dependencies"""
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

def create_annotated_image(image: Image.Image, prediction_result: dict) -> str:
    """Create annotated image with prediction results"""
    try:
        # Create a copy for annotation
        annotated_image = image.copy()
        draw = ImageDraw.Draw(annotated_image)
        
        # Try to load a font
        try:
            font_paths = [
                "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
                "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
                "/System/Library/Fonts/Arial.ttf",  # macOS
                "C:/Windows/Fonts/arial.ttf"  # Windows
            ]
            font = None
            for font_path in font_paths:
                try:
                    font = ImageFont.truetype(font_path, 20)
                    break
                except:
                    continue
            if font is None:
                font = ImageFont.load_default()
        except:
            font = ImageFont.load_default()
        
        # Get prediction info
        species = prediction_result['species']
        confidence = prediction_result['confidence_percentage']
        
        # Create text
        main_text = f"Detected: {species}"
        conf_text = f"Confidence: {confidence:.1f}%"
        
        # Position (top-left with padding)
        x, y = 10, 10
        
        # Choose color based on confidence
        if confidence >= 80:
            bg_color = "green"
        elif confidence >= 60:
            bg_color = "orange"
        else:
            bg_color = "red"
        
        # Draw background rectangles
        main_bbox = draw.textbbox((0, 0), main_text, font=font)
        conf_bbox = draw.textbbox((0, 0), conf_text, font=font)
        
        main_width = main_bbox[2] - main_bbox[0]
        main_height = main_bbox[3] - main_bbox[1]
        conf_width = conf_bbox[2] - conf_bbox[0]
        conf_height = conf_bbox[3] - conf_bbox[1]
        
        total_width = max(main_width, conf_width) + 20
        total_height = main_height + conf_height + 30
        
        # Draw background
        draw.rectangle([x, y, x + total_width, y + total_height], 
                      fill=bg_color, outline=bg_color)
        
        # Draw text
        draw.text((x + 10, y + 5), main_text, fill="white", font=font)
        draw.text((x + 10, y + main_height + 15), conf_text, fill="white", font=font)
        
        # Convert to base64
        buffer = BytesIO()
        annotated_image.save(buffer, format='JPEG', quality=95)
        img_str = base64.b64encode(buffer.getvalue()).decode()
        return img_str
        
    except Exception as e:
        logger.error(f"Error creating annotation: {e}")
        # Return original image as base64 if annotation fails
        buffer = BytesIO()
        image.save(buffer, format='JPEG', quality=95)
        img_str = base64.b64encode(buffer.getvalue()).decode()
        return img_str

@asynccontextmanager
async def lifespan(app: FastAPI):
    """TrueFoundry lifespan event handler"""
    logger.info("🚀 Starting application lifespan...")
    await load_models()
    logger.info("✅ Application startup complete!")
    yield
    logger.info("🔄 Application shutdown...")

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
    """Root endpoint with API information"""
    return {
        "message": "BioCLIP Pest Identification API",
        "version": "2.0.0",
        "status": "running",
        "model": "BioCLIP + Custom Classifier",
        "classes": CLASS_NAMES,
        "device": str(DEVICE),
        "endpoints": {
            "health": "/health",
            "predict": "/predict",
            "model_info": "/model-info",
            "docs": "/docs"
        }
    }

@app.get("/health")
async def health():
    """Health check endpoint for TrueFoundry readiness/liveness probes"""
    global bioclip_model, classifier
    
    try:
        # Check if models are loaded
        if bioclip_model is None or classifier is None:
            return JSONResponse(
                status_code=503,
                content={"healthy": False, "status": "models not loaded"}
            )
        
        # Quick model test
        dummy_input = torch.randn(1, 3, 224, 224).to(DEVICE)
        with torch.no_grad():
            features = bioclip_model.encode_image(dummy_input)
            outputs = classifier(features)
        
        return {
            "healthy": True, 
            "status": "ready", 
            "models_loaded": True,
            "device": str(DEVICE),
            "classes_count": len(CLASS_NAMES)
        }
        
    except Exception as e:
        return JSONResponse(
            status_code=503,
            content={"healthy": False, "status": f"health check failed: {str(e)}"}
        )

@app.get("/model-info")
async def model_info():
    """Get model information"""
    global bioclip_model, classifier
    
    return {
        "model_type": "BioCLIP + Custom Classifier",
        "foundation_model": "microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224",
        "custom_model_url": GITHUB_MODEL_URL,
        "device": str(DEVICE),
        "classes": CLASS_NAMES,
        "models_loaded": bioclip_model is not None and classifier is not None,
        "input_size": "224x224",
        "supported_formats": ["JPEG", "PNG", "BMP"]
    }

@app.post("/predict")
async def predict_pest(image: UploadFile = File(...)):
    """Main prediction endpoint for pest identification"""
    global bioclip_model, classifier
    
    # Check if models are loaded
    if bioclip_model is None or classifier is None:
        raise HTTPException(status_code=503, detail="Models not loaded yet")
    
    # Validate image
    if not image.content_type or not image.content_type.startswith('image/'):
        raise HTTPException(status_code=400, detail="File must be an image")
    
    # Check file size (10MB limit)
    if image.size and image.size > 10 * 1024 * 1024:
        raise HTTPException(status_code=400, detail="Image file too large (max 10MB)")
    
    try:
        # Read and process image
        logger.info(f"Processing image: {image.filename}")
        image_bytes = await image.read()
        pil_image = Image.open(BytesIO(image_bytes)).convert('RGB')
        
        # Preprocess image (no OpenCV/albumentations needed)
        transform = setup_image_transforms()
        image_tensor = transform(pil_image).unsqueeze(0).to(DEVICE)
        
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
        
        # Create prediction result
        prediction_result = {
            "species": predicted_class_name,
            "confidence": round(confidence_score, 4),
            "confidence_percentage": round(confidence_score * 100, 2)
        }
        
        # Create annotated image
        annotated_image_b64 = create_annotated_image(pil_image, prediction_result)
        
        # Format comprehensive response
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
            "image_info": {
                "filename": image.filename,
                "width": pil_image.size[0],
                "height": pil_image.size[1],
                "format": pil_image.format or "Unknown"
            },
            "annotated_image": annotated_image_b64,
            "processing_time": round(processing_time, 3),
            "device_used": str(DEVICE),
            "timestamp": time.time()
        }
        
        logger.info(f"Prediction completed: {predicted_class_name} ({confidence_score:.2%})")
        return JSONResponse(content=response)
        
    except Exception as e:
        logger.error(f"Prediction failed: {e}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

# Error handlers
@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    logger.error(f"Unhandled exception: {exc}")
    logger.error(f"Traceback: {traceback.format_exc()}")
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error", "type": type(exc).__name__}
    )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")
