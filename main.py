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
bioclip_preprocess = None
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

# Model Architecture - Simplified and more appropriate for BioCLIP
class OptimizedClassifier(nn.Module):
    def __init__(self, input_dim, num_classes, dropout_rate=0.3):
        super(OptimizedClassifier, self).__init__()
        self.classifier = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(input_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate * 0.5),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate * 0.3),
            nn.Linear(128, num_classes)
        )
        
        # Initialize weights
        self.apply(self._init_weights)
    
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                torch.nn.init.zeros_(m.bias)
    
    def forward(self, x):
        return self.classifier(x)

# Model Loading Functions
async def load_bioclip_models():
    global bioclip_model, bioclip_preprocess, classifier
    try:
        logger.info("Loading BioCLIP model...")
        
        # Load BioCLIP with proper preprocessing
        bioclip_model, _, bioclip_preprocess = open_clip.create_model_and_transforms(
            'hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224'
        )
        bioclip_model = bioclip_model.to(DEVICE)
        bioclip_model.eval()
        
        # Verify feature dimensions
        dummy_input = torch.randn(1, 3, 224, 224).to(DEVICE)
        with torch.no_grad():
            features = bioclip_model.encode_image(dummy_input)
            feature_dim = features.shape[-1]
        
        logger.info(f"BioCLIP loaded - Feature dimension: {feature_dim}")
        
        # Download classifier if needed
        if not os.path.exists(LOCAL_MODEL_PATH):
            await download_trained_classifier()
        
        # Load classifier
        classifier = OptimizedClassifier(feature_dim, len(CLASS_NAMES))
        
        # Load state dict with error handling
        try:
            state_dict = torch.load(LOCAL_MODEL_PATH, map_location=DEVICE)
            classifier.load_state_dict(state_dict)
        except Exception as e:
            logger.error(f"Failed to load classifier state dict: {e}")
            raise ValueError(f"Classifier model file is corrupted or incompatible: {e}")
        
        classifier = classifier.to(DEVICE)
        classifier.eval()
        logger.info("Trained classifier loaded successfully")
        
        # Test the complete pipeline
        with torch.no_grad():
            test_features = bioclip_model.encode_image(dummy_input)
            test_output = classifier(test_features)
            test_probs = F.softmax(test_output, dim=1)
            
            if test_probs.shape[1] != len(CLASS_NAMES):
                raise ValueError(f"Classifier output dimension {test_probs.shape[1]} doesn't match number of classes {len(CLASS_NAMES)}")
        
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
        
        total_size = int(response.headers.get('content-length', 0))
        downloaded = 0
        
        with open(LOCAL_MODEL_PATH, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
                    downloaded += len(chunk)
                    if total_size > 0:
                        progress = (downloaded / total_size) * 100
                        if downloaded % (1024 * 1024) == 0:  # Log every MB
                            logger.info(f"Download progress: {progress:.1f}%")
        
        logger.info(f"Classifier downloaded successfully ({downloaded} bytes)")
        
        # Verify file integrity
        if os.path.getsize(LOCAL_MODEL_PATH) < 1000:  # Minimum expected size
            raise ValueError("Downloaded file appears to be corrupted (too small)")
            
    except Exception as e:
        logger.error(f"Download failed: {e}")
        if os.path.exists(LOCAL_MODEL_PATH):
            os.remove(LOCAL_MODEL_PATH)
        raise

# Image Processing Functions
def preprocess_image(pil_image: Image.Image) -> torch.Tensor:
    """Preprocess image using BioCLIP's preprocessing pipeline"""
    global bioclip_preprocess
    
    if bioclip_preprocess is None:
        raise ValueError("BioCLIP preprocessing pipeline not loaded")
    
    try:
        # Use BioCLIP's preprocessing
        image_tensor = bioclip_preprocess(pil_image).unsqueeze(0)
        return image_tensor
    except Exception as e:
        logger.error(f"Image preprocessing failed: {e}")
        raise ValueError(f"Failed to preprocess image: {e}")

def validate_image(pil_image: Image.Image) -> bool:
    """Validate image properties"""
    if pil_image.mode not in ['RGB', 'RGBA']:
        return False
    
    width, height = pil_image.size
    if width < 32 or height < 32:
        return False
    
    if width > 4000 or height > 4000:
        return False
    
    return True

def create_prediction_visualization(image: Image.Image, prediction: dict) -> str:
    try:
        # Resize image if too large for visualization
        max_size = 800
        if max(image.size) > max_size:
            image.thumbnail((max_size, max_size), Image.Resampling.LANCZOS)
        
        annotated = image.copy()
        draw = ImageDraw.Draw(annotated)
        
        # Try to load a better font
        try:
            font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 20)
        except:
            try:
                font = ImageFont.truetype("arial.ttf", 20)
            except:
                font = ImageFont.load_default()
        
        species = prediction['species']
        confidence = prediction['confidence_percentage']
        main_text = f"Species: {species}"
        conf_text = f"Confidence: {confidence:.1f}%"
        
        # Color coding based on confidence
        if confidence >= 90:
            bg_color, text_color = "#27AE60", "white"
        elif confidence >= 70:
            bg_color, text_color = "#F39C12", "white"
        elif confidence >= 50:
            bg_color, text_color = "#E67E22", "white"
        else:
            bg_color, text_color = "#E74C3C", "white"
        
        x, y = 10, 10
        padding = 10
        
        # Calculate text dimensions
        main_bbox = draw.textbbox((0, 0), main_text, font=font)
        conf_bbox = draw.textbbox((0, 0), conf_text, font=font)
        
        width = max(main_bbox[2] - main_bbox[0], conf_bbox[2] - conf_bbox[0]) + 2 * padding
        height = (main_bbox[3] - main_bbox[1]) + (conf_bbox[3] - conf_bbox[1]) + 3 * padding
        
        # Draw background rectangle
        draw.rectangle([x, y, x + width, y + height], fill=bg_color, outline="black", width=2)
        
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
        # Return original image if visualization fails
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
    version="3.1.0",
    description="API for pest identification using BioCLIP with improved preprocessing and error handling",
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
        "version": "3.1.0",
        "status": "running",
        "model_info": {
            "foundation_model": "BiomedCLIP-PubMedBERT_256-vit_base_patch16_224",
            "classes": CLASS_NAMES,
            "device": str(DEVICE),
            "num_classes": len(CLASS_NAMES)
        }
    }

@app.get("/health")
async def health_check():
    global bioclip_model, classifier, bioclip_preprocess
    
    if any(model is None for model in [bioclip_model, classifier, bioclip_preprocess]):
        return JSONResponse(
            status_code=503,
            content={"healthy": False, "status": "BioCLIP models not loaded"}
        )
    
    try:
        # Test the complete pipeline
        test_input = torch.randn(1, 3, 224, 224).to(DEVICE)
        with torch.no_grad():
            features = bioclip_model.encode_image(test_input)
            outputs = classifier(features)
            probabilities = F.softmax(outputs, dim=1)
            
        return {
            "healthy": True, 
            "status": "ready", 
            "device": str(DEVICE),
            "feature_dim": features.shape[-1],
            "num_classes": probabilities.shape[-1]
        }
    except Exception as e:
        return JSONResponse(
            status_code=503,
            content={"healthy": False, "error": str(e)}
        )

@app.post("/predict")
async def predict_pest(image: UploadFile = File(...)):
    global bioclip_model, classifier, bioclip_preprocess
    
    if any(model is None for model in [bioclip_model, classifier, bioclip_preprocess]):
        raise HTTPException(status_code=503, detail="BioCLIP models not loaded")
    
    if not image.content_type.startswith('image/'):
        raise HTTPException(status_code=400, detail="File must be an image")
    
    try:
        # Read and validate image
        image_bytes = await image.read()
        if len(image_bytes) == 0:
            raise HTTPException(status_code=400, detail="Empty image file")
        
        if len(image_bytes) > 10 * 1024 * 1024:  # 10MB limit
            raise HTTPException(status_code=400, detail="Image file too large (max 10MB)")
        
        pil_image = Image.open(BytesIO(image_bytes)).convert('RGB')
        
        if not validate_image(pil_image):
            raise HTTPException(status_code=400, detail="Invalid image dimensions or format")
        
        # Preprocess image using BioCLIP's preprocessing
        image_tensor = preprocess_image(pil_image).to(DEVICE)
        
        start_time = time.time()
        with torch.no_grad():
            # Extract features using BioCLIP
            features = bioclip_model.encode_image(image_tensor)
            
            # Classify using trained classifier
            outputs = classifier(features)
            probabilities = F.softmax(outputs, dim=1)
            
            # Get prediction
            confidence, predicted_class = torch.max(probabilities, 1)
            predicted_idx = predicted_class.item()
            predicted_species = CLASS_NAMES[predicted_idx]
            
            # Get all class probabilities for additional insight
            all_probs = probabilities.cpu().numpy().flatten()
            class_scores = {class_name: float(prob) for class_name, prob in zip(CLASS_NAMES, all_probs)}
        
        processing_time = time.time() - start_time
        
        prediction_result = {
            "species": predicted_species,
            "confidence": round(confidence.item(), 4),
            "confidence_percentage": round(confidence.item() * 100, 2),
            "all_class_scores": class_scores
        }
        
        # Create visualization
        annotated_image = create_prediction_visualization(pil_image, prediction_result)
        
        return {
            "success": True,
            "prediction": prediction_result,
            "annotated_image": annotated_image,
            "processing_time": round(processing_time, 3),
            "device": str(DEVICE),
            "image_size": pil_image.size
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Prediction failed: {e}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

@app.get("/classes")
async def get_classes():
    """Get available pest classes"""
    return {
        "classes": CLASS_NAMES,
        "num_classes": len(CLASS_NAMES)
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=False)
