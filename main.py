import os
import sys
import requests
import torch
import torch.nn as nn
import torch.nn.functional as F
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
import traceback

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

# Model Architecture
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
        self.apply(self._init_weights)
    
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                torch.nn.init.zeros_(m.bias)
    
    def forward(self, x):
        return self.classifier(x)

def check_dependencies():
    """Check if all required dependencies are installed"""
    required_packages = {
        'open_clip': 'open_clip_torch',
        'transformers': 'transformers',
        'torch': 'torch',
        'torchvision': 'torchvision',
        'PIL': 'Pillow',
        'requests': 'requests'
    }
    
    missing_packages = []
    for package, pip_name in required_packages.items():
        try:
            __import__(package)
            logger.info(f"✓ {package} is available")
        except ImportError:
            missing_packages.append(pip_name)
            logger.error(f"✗ {package} is missing")
    
    if missing_packages:
        logger.error(f"Missing packages: {missing_packages}")
        logger.error("Install with: pip install " + " ".join(missing_packages))
        return False
    return True

def get_system_info():
    """Get system information for debugging"""
    info = {
        "python_version": sys.version,
        "torch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_device_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
        "device": str(DEVICE)
    }
    
    if torch.cuda.is_available():
        info["cuda_device_name"] = torch.cuda.get_device_name(0)
        info["cuda_memory_total"] = torch.cuda.get_device_properties(0).total_memory
    
    # Fixed: Handle open_clip version properly
    try:
        import open_clip
        # Try different ways to get version
        version = getattr(open_clip, '__version__', None)
        if version is None:
            # Try to get version from package metadata
            try:
                import pkg_resources
                version = pkg_resources.get_distribution('open_clip_torch').version
            except:
                version = "Available but version unknown"
        info["open_clip_version"] = version
    except ImportError:
        info["open_clip_version"] = "Not available"
    except Exception as e:
        info["open_clip_version"] = f"Error: {str(e)}"
    
    return info

async def load_bioclip_models():
    """Load BioCLIP models with comprehensive error handling"""
    global bioclip_model, bioclip_preprocess, classifier
    
    try:
        # Check dependencies first
        if not check_dependencies():
            raise RuntimeError("Missing required dependencies")
        
        # Log system information
        system_info = get_system_info()
        logger.info(f"System Info: {system_info}")
        
        # Import open_clip after dependency check
        try:
            import open_clip
            logger.info("✓ open_clip imported successfully")
        except Exception as e:
            logger.error(f"Failed to import open_clip: {e}")
            raise RuntimeError(f"open_clip import failed: {e}")
        
        logger.info("Loading BioCLIP model...")
        
        # Method 1: Try standard loading with error handling
        try:
            bioclip_model, _, bioclip_preprocess = open_clip.create_model_and_transforms(
                'hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224'
            )
            logger.info("✓ BioCLIP loaded successfully with standard method")
        except Exception as e1:
            logger.warning(f"Standard loading failed: {e1}")
            
            # Method 2: Try with cache directory
            try:
                os.makedirs('./model_cache', exist_ok=True)
                bioclip_model, _, bioclip_preprocess = open_clip.create_model_and_transforms(
                    'hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224',
                    device=DEVICE,
                    cache_dir='./model_cache'
                )
                logger.info("✓ BioCLIP loaded successfully with cache method")
            except Exception as e2:
                logger.warning(f"Cache loading failed: {e2}")
                
                # Method 3: Try alternative loading approach
                try:
                    bioclip_model = open_clip.create_model(
                        'hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224'
                    )
                    _, bioclip_preprocess = open_clip.create_model_and_transforms(
                        'hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224'
                    )[1:]
                    logger.info("✓ BioCLIP loaded successfully with alternative method")
                except Exception as e3:
                    logger.error(f"All loading methods failed: {e1}, {e2}, {e3}")
                    raise RuntimeError(f"Failed to load BioCLIP model: {e3}")
        
        # Move model to device and set eval mode
        bioclip_model = bioclip_model.to(DEVICE)
        bioclip_model.eval()
        logger.info(f"BioCLIP model moved to device: {DEVICE}")
        
        # Test model with dummy input
        try:
            dummy_input = torch.randn(1, 3, 224, 224).to(DEVICE)
            with torch.no_grad():
                features = bioclip_model.encode_image(dummy_input)
                feature_dim = features.shape[-1]
            logger.info(f"✓ BioCLIP test successful - Feature dimension: {feature_dim}")
        except Exception as e:
            logger.error(f"BioCLIP model test failed: {e}")
            raise RuntimeError(f"BioCLIP model test failed: {e}")
        
        # Download classifier if needed
        if not os.path.exists(LOCAL_MODEL_PATH):
            logger.info("Classifier not found locally, downloading...")
            try:
                await download_trained_classifier()
            except Exception as e:
                logger.error(f"Failed to download classifier: {e}")
                raise RuntimeError(f"Failed to download classifier: {e}")
        else:
            logger.info("Using local classifier file")
        
        # Load classifier
        try:
            classifier = OptimizedClassifier(feature_dim, len(CLASS_NAMES))
            
            # Load state dict with error handling
            state_dict = torch.load(LOCAL_MODEL_PATH, map_location=DEVICE)
            
            # Check if state dict has the expected keys
            model_keys = set(classifier.state_dict().keys())
            loaded_keys = set(state_dict.keys())
            
            if model_keys != loaded_keys:
                logger.warning(f"State dict keys mismatch:")
                logger.warning(f"Model keys: {model_keys}")
                logger.warning(f"Loaded keys: {loaded_keys}")
                
                # Try to load with strict=False
                classifier.load_state_dict(state_dict, strict=False)
                logger.info("✓ Classifier loaded with strict=False")
            else:
                classifier.load_state_dict(state_dict)
                logger.info("✓ Classifier loaded successfully")
            
            classifier = classifier.to(DEVICE)
            classifier.eval()
            
        except Exception as e:
            logger.error(f"Classifier loading failed: {e}")
            raise RuntimeError(f"Classifier loading failed: {e}")
        
        # Test complete pipeline
        try:
            with torch.no_grad():
                test_features = bioclip_model.encode_image(dummy_input)
                test_output = classifier(test_features)
                test_probs = F.softmax(test_output, dim=1)
                
                if test_probs.shape[1] != len(CLASS_NAMES):
                    raise ValueError(f"Output dimension mismatch: {test_probs.shape[1]} vs {len(CLASS_NAMES)}")
                
                logger.info(f"✓ Complete pipeline test successful")
                logger.info(f"  - Feature shape: {test_features.shape}")
                logger.info(f"  - Output shape: {test_output.shape}")
                logger.info(f"  - Probability shape: {test_probs.shape}")
        except Exception as e:
            logger.error(f"Pipeline test failed: {e}")
            raise RuntimeError(f"Pipeline test failed: {e}")
        
        logger.info("🎉 All models loaded successfully!")
        
    except Exception as e:
        logger.error(f"Model loading failed: {e}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        # Reset global variables
        bioclip_model = None
        bioclip_preprocess = None
        classifier = None
        raise

async def download_trained_classifier():
    """Download classifier with progress tracking and verification"""
    try:
        logger.info("Downloading trained classifier...")
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(LOCAL_MODEL_PATH) if os.path.dirname(LOCAL_MODEL_PATH) else '.', exist_ok=True)
        
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        
        # Use session for better connection handling
        session = requests.Session()
        session.headers.update(headers)
        
        response = session.get(GITHUB_MODEL_URL, stream=True, timeout=300)
        response.raise_for_status()
        
        total_size = int(response.headers.get('content-length', 0))
        downloaded = 0
        
        with open(LOCAL_MODEL_PATH, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
                    downloaded += len(chunk)
                    
                    # Log progress every 5MB
                    if total_size > 0 and downloaded % (5 * 1024 * 1024) == 0:
                        progress = (downloaded / total_size) * 100
                        logger.info(f"Download progress: {progress:.1f}% ({downloaded}/{total_size} bytes)")
        
        logger.info(f"✓ Classifier downloaded successfully ({downloaded} bytes)")
        
        # Verify file integrity
        file_size = os.path.getsize(LOCAL_MODEL_PATH)
        if file_size < 1000:  # Minimum expected size
            raise ValueError(f"Downloaded file appears corrupted (size: {file_size} bytes)")
        
        # Try to load the file to verify it's a valid torch model
        try:
            torch.load(LOCAL_MODEL_PATH, map_location='cpu')
            logger.info("✓ Downloaded model file verification successful")
        except Exception as e:
            raise ValueError(f"Downloaded model file is corrupted: {e}")
            
    except Exception as e:
        logger.error(f"Download failed: {e}")
        # Clean up corrupted file
        if os.path.exists(LOCAL_MODEL_PATH):
            os.remove(LOCAL_MODEL_PATH)
        raise

def preprocess_image(pil_image: Image.Image) -> torch.Tensor:
    """Preprocess image with fallback options"""
    global bioclip_preprocess
    
    if bioclip_preprocess is None:
        # Fallback preprocessing if BioCLIP preprocess is not available
        import torchvision.transforms as transforms
        fallback_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        logger.warning("Using fallback preprocessing")
        return fallback_transform(pil_image).unsqueeze(0)
    
    try:
        return bioclip_preprocess(pil_image).unsqueeze(0)
    except Exception as e:
        logger.error(f"BioCLIP preprocessing failed: {e}")
        # Fallback to manual preprocessing
        import torchvision.transforms as transforms
        fallback_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        logger.warning("Using fallback preprocessing due to error")
        return fallback_transform(pil_image).unsqueeze(0)

def create_prediction_visualization(image: Image.Image, prediction: dict) -> str:
    """Create visualization with error handling"""
    try:
        # Resize large images
        max_size = 800
        if max(image.size) > max_size:
            image.thumbnail((max_size, max_size), Image.Resampling.LANCZOS)
        
        annotated = image.copy()
        draw = ImageDraw.Draw(annotated)
        
        # Try to load fonts
        font = None
        for font_path in [
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
            "/System/Library/Fonts/Arial.ttf",
            "arial.ttf",
            "C:/Windows/Fonts/arial.ttf"
        ]:
            try:
                font = ImageFont.truetype(font_path, 20)
                break
            except:
                continue
        
        if font is None:
            font = ImageFont.load_default()
        
        species = prediction['species']
        confidence = prediction['confidence_percentage']
        main_text = f"Species: {species}"
        conf_text = f"Confidence: {confidence:.1f}%"
        
        # Color coding
        if confidence >= 90:
            bg_color, text_color = "#27AE60", "white"
        elif confidence >= 70:
            bg_color, text_color = "#F39C12", "white"
        elif confidence >= 50:
            bg_color, text_color = "#E67E22", "white"
        else:
            bg_color, text_color = "#E74C3C", "white"
        
        # Draw annotation
        x, y = 10, 10
        padding = 10
        
        main_bbox = draw.textbbox((0, 0), main_text, font=font)
        conf_bbox = draw.textbbox((0, 0), conf_text, font=font)
        
        width = max(main_bbox[2] - main_bbox[0], conf_bbox[2] - conf_bbox[0]) + 2 * padding
        height = (main_bbox[3] - main_bbox[1]) + (conf_bbox[3] - conf_bbox[1]) + 3 * padding
        
        draw.rectangle([x, y, x + width, y + height], fill=bg_color, outline="black", width=2)
        draw.text((x + padding, y + padding), main_text, fill=text_color, font=font)
        draw.text((x + padding, y + padding + main_bbox[3] - main_bbox[1] + padding), 
                 conf_text, fill=text_color, font=font)
        
        # Convert to base64
        buffer = BytesIO()
        annotated.save(buffer, format='JPEG', quality=95)
        return base64.b64encode(buffer.getvalue()).decode()
        
    except Exception as e:
        logger.error(f"Visualization error: {e}")
        # Return original image
        buffer = BytesIO()
        image.save(buffer, format='JPEG', quality=95)
        return base64.b64encode(buffer.getvalue()).decode()

# FastAPI Application with improved error handling
@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("=" * 50)
    logger.info("Starting BioCLIP API Server...")
    logger.info("=" * 50)
    
    startup_success = False
    try:
        await load_bioclip_models()
        startup_success = True
        logger.info("🚀 BioCLIP API Server ready!")
    except Exception as e:
        logger.error(f"❌ Startup failed: {e}")
        logger.error("Server will start but models are not available")
    
    yield
    
    logger.info("Shutting down BioCLIP API Server...")

app = FastAPI(
    title="BioCLIP Pest Identification API",
    version="3.2.1",
    description="API for pest identification using BioCLIP with comprehensive error handling",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
async def root():
    global bioclip_model, classifier
    model_status = "loaded" if bioclip_model is not None and classifier is not None else "not_loaded"
    
    return {
        "message": "BioCLIP Pest Identification API",
        "version": "3.2.1",
        "status": "running",
        "model_status": model_status,
        "model_info": {
            "foundation_model": "BiomedCLIP-PubMedBERT_256-vit_base_patch16_224",
            "classes": CLASS_NAMES,
            "device": str(DEVICE),
            "num_classes": len(CLASS_NAMES)
        },
        "system_info": get_system_info()
    }

@app.get("/health")
async def health_check():
    global bioclip_model, classifier, bioclip_preprocess
    
    # Check if models are loaded
    models_loaded = all(model is not None for model in [bioclip_model, classifier])
    
    if not models_loaded:
        return JSONResponse(
            status_code=503,
            content={
                "healthy": False, 
                "status": "models_not_loaded",
                "bioclip_model": bioclip_model is not None,
                "classifier": classifier is not None,
                "preprocess": bioclip_preprocess is not None,
                "system_info": get_system_info()
            }
        )
    
    # Test models
    try:
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
            "num_classes": probabilities.shape[-1],
            "system_info": get_system_info()
        }
    except Exception as e:
        return JSONResponse(
            status_code=503,
            content={
                "healthy": False,
                "status": "model_test_failed",
                "error": str(e),
                "system_info": get_system_info()
            }
        )

@app.post("/predict")
async def predict_pest(image: UploadFile = File(...)):
    global bioclip_model, classifier, bioclip_preprocess
    
    # Check if models are loaded
    if any(model is None for model in [bioclip_model, classifier]):
        raise HTTPException(
            status_code=503, 
            detail="BioCLIP models not loaded. Check /health endpoint for details."
        )
    
    # Validate file
    if not image.content_type.startswith('image/'):
        raise HTTPException(status_code=400, detail="File must be an image")
    
    try:
        # Read and process image
        image_bytes = await image.read()
        if len(image_bytes) == 0:
            raise HTTPException(status_code=400, detail="Empty image file")
        
        if len(image_bytes) > 10 * 1024 * 1024:  # 10MB limit
            raise HTTPException(status_code=400, detail="Image too large (max 10MB)")
        
        pil_image = Image.open(BytesIO(image_bytes)).convert('RGB')
        
        # Validate image dimensions
        if min(pil_image.size) < 32:
            raise HTTPException(status_code=400, detail="Image too small (min 32x32)")
        
        # Preprocess image
        image_tensor = preprocess_image(pil_image).to(DEVICE)
        
        # Make prediction
        start_time = time.time()
        with torch.no_grad():
            features = bioclip_model.encode_image(image_tensor)
            outputs = classifier(features)
            probabilities = F.softmax(outputs, dim=1)
            
            confidence, predicted_class = torch.max(probabilities, 1)
            predicted_idx = predicted_class.item()
            predicted_species = CLASS_NAMES[predicted_idx]
            
            # Get all class scores
            all_probs = probabilities.cpu().numpy().flatten()
            class_scores = {class_name: float(prob) for class_name, prob in zip(CLASS_NAMES, all_probs)}
        
        processing_time = time.time() - start_time
        
        # Create prediction result
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
        logger.error(f"Prediction error: {e}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

@app.get("/debug")
async def debug_info():
    """Debug endpoint to check system state"""
    global bioclip_model, classifier, bioclip_preprocess
    
    return {
        "dependencies": check_dependencies(),
        "system_info": get_system_info(),
        "model_status": {
            "bioclip_model": bioclip_model is not None,
            "classifier": classifier is not None,
            "preprocess": bioclip_preprocess is not None
        },
        "files": {
            "classifier_exists": os.path.exists(LOCAL_MODEL_PATH),
            "classifier_size": os.path.getsize(LOCAL_MODEL_PATH) if os.path.exists(LOCAL_MODEL_PATH) else 0
        },
        "device_info": {
            "device": str(DEVICE),
            "cuda_available": torch.cuda.is_available(),
            "cuda_device_count": torch.cuda.device_count() if torch.cuda.is_available() else 0
        }
    }

@app.post("/reload")
async def reload_models():
    """Manually reload models"""
    try:
        logger.info("Manual model reload requested...")
        await load_bioclip_models()
        return {"success": True, "message": "Models reloaded successfully"}
    except Exception as e:
        logger.error(f"Manual reload failed: {e}")
        return JSONResponse(
            status_code=500,
            content={"success": False, "error": str(e)}
        )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=False)
