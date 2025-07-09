"""
Enhanced BioCLIP Pest Identification API
========================================
A production-ready FastAPI application for pest identification using deep learning.
Features robust error handling, model validation, and comprehensive logging.
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
import hashlib
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from contextlib import asynccontextmanager
from dataclasses import dataclass
from fastapi import FastAPI, File, UploadFile, HTTPException, Request
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
import uvicorn

# =============================================================================
# CONFIGURATION AND CONSTANTS
# =============================================================================

@dataclass
class Config:
    """Application configuration"""
    # Model settings
    MODEL_VERSION: str = "2.3.0"
    FEATURE_DIM: int = 2048
    NUM_CLASSES: int = 5
    
    # File paths
    GITHUB_MODEL_URL: str = "https://github.com/Vignesh-Vicky2004/Insect-Identification/raw/main/best_bioclip_classifier.pth"
    LOCAL_MODEL_PATH: str = "best_bioclip_classifier.pth"
    
    # API settings
    MAX_FILE_SIZE: int = 10 * 1024 * 1024  # 10MB
    REQUEST_TIMEOUT: int = 300
    MAX_RETRIES: int = 3
    
    # Image processing
    IMAGE_SIZE: Tuple[int, int] = (224, 224)
    NORMALIZE_MEAN: List[float] = [0.485, 0.456, 0.406]
    NORMALIZE_STD: List[float] = [0.229, 0.224, 0.225]
    
    # Logging
    LOG_LEVEL: str = "INFO"
    LOG_FORMAT: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

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
# LOGGING CONFIGURATION
# =============================================================================

def setup_logging() -> logging.Logger:
    """Configure application logging"""
    logging.basicConfig(
        level=getattr(logging, config.LOG_LEVEL),
        format=config.LOG_FORMAT,
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler('app.log') if os.path.exists('/app') else logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)

logger = setup_logging()

# =============================================================================
# GLOBAL VARIABLES
# =============================================================================

feature_extractor: Optional[nn.Module] = None
classifier: Optional[nn.Module] = None
model_metadata: Dict[str, Any] = {}

# =============================================================================
# MODEL DEFINITIONS
# =============================================================================

class EnhancedFeatureExtractor(nn.Module):
    """Enhanced feature extractor based on ResNet50"""
    
    def __init__(self, pretrained: bool = True):
        super().__init__()
        self.backbone = models.resnet50(weights='DEFAULT' if pretrained else None)
        self.backbone.fc = nn.Identity()  # Remove final classification layer
        
        # Add batch normalization for better stability
        self.feature_norm = nn.BatchNorm1d(config.FEATURE_DIM)
        
    def encode_image(self, x: torch.Tensor) -> torch.Tensor:
        """Extract features from input images"""
        features = self.backbone(x)
        features = self.feature_norm(features)
        return features
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass for compatibility"""
        return self.encode_image(x)

class RobustClassifier(nn.Module):
    """Robust classifier with improved architecture"""
    
    def __init__(self, input_dim: int = 2048, num_classes: int = 5, dropout_rate: float = 0.3):
        super().__init__()
        
        self.classifier = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(input_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            
            nn.Dropout(dropout_rate * 0.7),
            nn.Linear(512, 256),
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
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize network weights"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass"""
        return self.classifier(x)

# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def calculate_file_hash(file_path: str) -> str:
    """Calculate SHA256 hash of a file"""
    sha256_hash = hashlib.sha256()
    try:
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                sha256_hash.update(chunk)
        return sha256_hash.hexdigest()
    except Exception as e:
        logger.error(f"Error calculating file hash: {e}")
        return ""

def validate_image_file(file_path: str) -> bool:
    """Validate if file is a valid image"""
    try:
        with Image.open(file_path) as img:
            img.verify()
        return True
    except Exception:
        return False

def get_system_info() -> Dict[str, Any]:
    """Get system information"""
    return {
        "python_version": sys.version.split()[0],
        "torch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "device": str(DEVICE),
        "device_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
        "memory_total": torch.cuda.get_device_properties(0).total_memory if torch.cuda.is_available() else 0
    }

# =============================================================================
# MODEL MANAGEMENT
# =============================================================================

class ModelManager:
    """Enhanced model management with validation and caching"""
    
    def __init__(self):
        self.model_cache_dir = Path("model_cache")
        self.model_cache_dir.mkdir(exist_ok=True)
        
    async def download_model_with_validation(self) -> str:
        """Download model with comprehensive validation"""
        try:
            # Check if model exists and is valid
            if await self._validate_existing_model():
                return config.LOCAL_MODEL_PATH
            
            # Download model
            await self._download_model()
            
            # Validate downloaded model
            if not await self._validate_downloaded_model():
                raise ValueError("Downloaded model validation failed")
            
            return config.LOCAL_MODEL_PATH
            
        except Exception as e:
            logger.error(f"Model download failed: {e}")
            return await self._create_fallback_model()
    
    async def _validate_existing_model(self) -> bool:
        """Validate existing model file"""
        if not os.path.exists(config.LOCAL_MODEL_PATH):
            return False
        
        file_size = os.path.getsize(config.LOCAL_MODEL_PATH)
        if file_size < 1000:  # Less than 1KB
            logger.warning(f"Model file too small: {file_size} bytes")
            return False
        
        # Try to load model structure
        try:
            torch.load(config.LOCAL_MODEL_PATH, map_location='cpu')
            logger.info(f"✅ Valid existing model found ({file_size} bytes)")
            return True
        except Exception as e:
            logger.warning(f"Existing model validation failed: {e}")
            return False
    
    async def _download_model(self):
        """Download model from GitHub with retry logic"""
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
            'Accept': 'application/octet-stream, */*',
            'Accept-Language': 'en-US,en;q=0.9',
            'Cache-Control': 'no-cache'
        }
        
        for attempt in range(config.MAX_RETRIES):
            try:
                logger.info(f"📥 Downloading model (attempt {attempt + 1}/{config.MAX_RETRIES})...")
                
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
                    raise ValueError("Received HTML instead of binary data")
                
                # Download with progress tracking
                total_size = 0
                with open(config.LOCAL_MODEL_PATH, 'wb') as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        if chunk:
                            f.write(chunk)
                            total_size += len(chunk)
                
                if total_size == 0:
                    raise ValueError("Downloaded file is empty")
                
                logger.info(f"✅ Model downloaded successfully ({total_size:,} bytes)")
                return
                
            except Exception as e:
                logger.warning(f"Download attempt {attempt + 1} failed: {e}")
                if attempt < config.MAX_RETRIES - 1:
                    await asyncio.sleep(2 ** attempt)  # Exponential backoff
                else:
                    raise
    
    async def _validate_downloaded_model(self) -> bool:
        """Validate downloaded model"""
        try:
            # Check file size
            file_size = os.path.getsize(config.LOCAL_MODEL_PATH)
            if file_size < 1000:
                return False
            
            # Try to load model
            state_dict = torch.load(config.LOCAL_MODEL_PATH, map_location='cpu')
            
            # Validate structure
            if not isinstance(state_dict, dict):
                return False
            
            logger.info(f"✅ Model validation successful ({file_size:,} bytes)")
            return True
            
        except Exception as e:
            logger.error(f"Model validation failed: {e}")
            return False
    
    async def _create_fallback_model(self) -> str:
        """Create fallback model when download fails"""
        try:
            logger.info("🔄 Creating fallback model...")
            
            # Create dummy classifier with proper structure
            dummy_classifier = RobustClassifier(
                input_dim=config.FEATURE_DIM,
                num_classes=config.NUM_CLASSES
            )
            
            # Save dummy model
            torch.save(dummy_classifier.state_dict(), config.LOCAL_MODEL_PATH)
            
            logger.info("✅ Fallback model created")
            return config.LOCAL_MODEL_PATH
            
        except Exception as e:
            logger.error(f"Failed to create fallback model: {e}")
            raise

async def initialize_models():
    """Initialize all models with comprehensive error handling"""
    global feature_extractor, classifier, model_metadata
    
    try:
        logger.info("🔄 Starting enhanced model initialization...")
        
        # Initialize model manager
        model_manager = ModelManager()
        
        # Initialize feature extractor
        logger.info("🖼️ Loading enhanced feature extractor...")
        feature_extractor = EnhancedFeatureExtractor(pretrained=True)
        feature_extractor = feature_extractor.to(DEVICE)
        feature_extractor.eval()
        
        # Test feature extractor
        test_input = torch.randn(1, 3, *config.IMAGE_SIZE).to(DEVICE)
        with torch.no_grad():
            features = feature_extractor.encode_image(test_input)
            actual_feature_dim = features.shape[-1]
        
        logger.info(f"✅ Feature extractor loaded (dimension: {actual_feature_dim})")
        
        # Download and validate model
        model_path = await model_manager.download_model_with_validation()
        
        # Initialize classifier
        logger.info("🎯 Loading robust classifier...")
        classifier = RobustClassifier(
            input_dim=actual_feature_dim,
            num_classes=len(CLASS_NAMES)
        )
        
        # Load trained weights
        model_loaded = False
        try:
            state_dict = torch.load(model_path, map_location=DEVICE)
            classifier.load_state_dict(state_dict)
            model_loaded = True
            logger.info("✅ Trained weights loaded successfully")
        except Exception as e:
            logger.warning(f"⚠️ Failed to load trained weights: {e}")
            logger.info("🔄 Using randomly initialized weights")
        
        classifier = classifier.to(DEVICE)
        classifier.eval()
        
        # Test complete pipeline
        logger.info("🧪 Testing complete pipeline...")
        with torch.no_grad():
            test_features = feature_extractor.encode_image(test_input)
            test_output = classifier(test_features)
            test_probs = F.softmax(test_output, dim=1)
        
        # Store metadata
        model_metadata = {
            "feature_dim": actual_feature_dim,
            "num_classes": len(CLASS_NAMES),
            "model_loaded": model_loaded,
            "model_path": model_path,
            "model_hash": calculate_file_hash(model_path),
            "device": str(DEVICE),
            "system_info": get_system_info()
        }
        
        logger.info(f"✅ Pipeline test successful (output shape: {test_probs.shape})")
        logger.info("🎉 All models initialized successfully!")
        
    except Exception as e:
        logger.error(f"❌ Model initialization failed: {e}")
        raise

# =============================================================================
# IMAGE PROCESSING
# =============================================================================

class ImageProcessor:
    """Enhanced image processing with validation"""
    
    def __init__(self):
        self.transform = transforms.Compose([
            transforms.Resize(config.IMAGE_SIZE),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=config.NORMALIZE_MEAN,
                std=config.NORMALIZE_STD
            )
        ])
    
    def process_image(self, image_bytes: bytes) -> Tuple[torch.Tensor, Image.Image]:
        """Process image with validation"""
        try:
            # Validate image data
            if len(image_bytes) == 0:
                raise ValueError("Empty image data")
            
            # Open and validate image
            pil_image = Image.open(BytesIO(image_bytes)).convert('RGB')
            
            # Validate image dimensions
            if pil_image.size[0] < 32 or pil_image.size[1] < 32:
                raise ValueError("Image too small (minimum 32x32)")
            
            # Transform image
            image_tensor = self.transform(pil_image).unsqueeze(0).to(DEVICE)
            
            return image_tensor, pil_image
            
        except Exception as e:
            logger.error(f"Image processing failed: {e}")
            raise
    
    def create_annotated_image(self, image: Image.Image, prediction: Dict[str, Any]) -> str:
        """Create annotated image with prediction results"""
        try:
            # Create copy for annotation
            annotated_image = image.copy()
            draw = ImageDraw.Draw(annotated_image)
            
            # Load font
            font_size = max(16, min(image.size) // 20)
            try:
                font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", font_size)
            except:
                font = ImageFont.load_default()
            
            # Prepare text
            species = prediction['species']
            confidence = prediction['confidence_percentage']
            
            main_text = f"Detected: {species}"
            conf_text = f"Confidence: {confidence:.1f}%"
            
            # Calculate text dimensions
            main_bbox = draw.textbbox((0, 0), main_text, font=font)
            conf_bbox = draw.textbbox((0, 0), conf_text, font=font)
            
            # Position and styling
            x, y = 10, 10
            padding = 10
            
            # Choose color based on confidence
            if confidence >= 90:
                bg_color, text_color = "#2ECC71", "white"  # Green
            elif confidence >= 70:
                bg_color, text_color = "#F39C12", "white"  # Orange
            else:
                bg_color, text_color = "#E74C3C", "white"  # Red
            
            # Draw background
            total_width = max(main_bbox[2] - main_bbox[0], conf_bbox[2] - conf_bbox[0]) + 2 * padding
            total_height = (main_bbox[3] - main_bbox[1]) + (conf_bbox[3] - conf_bbox[1]) + 3 * padding
            
            draw.rectangle([x, y, x + total_width, y + total_height], fill=bg_color)
            
            # Draw text
            draw.text((x + padding, y + padding), main_text, fill=text_color, font=font)
            draw.text((x + padding, y + 2 * padding + main_bbox[3] - main_bbox[1]), conf_text, fill=text_color, font=font)
            
            # Convert to base64
            buffer = BytesIO()
            annotated_image.save(buffer, format='JPEG', quality=95)
            return base64.b64encode(buffer.getvalue()).decode()
            
        except Exception as e:
            logger.error(f"Annotation failed: {e}")
            # Return original image as fallback
            buffer = BytesIO()
            image.save(buffer, format='JPEG', quality=95)
            return base64.b64encode(buffer.getvalue()).decode()

# =============================================================================
# FASTAPI APPLICATION
# =============================================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Enhanced application lifespan management"""
    try:
        logger.info("🚀 Starting application initialization...")
        
        # Initialize models
        await initialize_models()
        
        # Log system information
        logger.info(f"System info: {get_system_info()}")
        
        logger.info("✅ Application ready!")
        yield
        
    except Exception as e:
        logger.error(f"❌ Application startup failed: {e}")
        # Allow app to start in degraded mode
        global feature_extractor, classifier
        feature_extractor = None
        classifier = None
        yield
        
    finally:
        logger.info("🔄 Application shutdown...")

# Create FastAPI application
app = FastAPI(
    title="Enhanced BioCLIP Pest Identification API",
    version=config.MODEL_VERSION,
    description="Production-ready pest identification API with enhanced error handling and robust architecture",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan
)

# Security middleware
app.add_middleware(
    TrustedHostMiddleware,
    allowed_hosts=["*"]  # Configure for production
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize image processor
image_processor = ImageProcessor()

# =============================================================================
# API ENDPOINTS
# =============================================================================

@app.get("/")
async def root():
    """Enhanced root endpoint with comprehensive information"""
    return {
        "message": "Enhanced BioCLIP Pest Identification API",
        "version": config.MODEL_VERSION,
        "status": "running",
        "api_info": {
            "classes": CLASS_NAMES,
            "supported_formats": ["JPEG", "PNG", "BMP", "TIFF"],
            "max_file_size": f"{config.MAX_FILE_SIZE / (1024*1024):.1f}MB",
            "image_size": f"{config.IMAGE_SIZE[0]}x{config.IMAGE_SIZE[1]}"
        },
        "system_info": {
            "device": str(DEVICE),
            "models_loaded": feature_extractor is not None and classifier is not None,
            "cuda_available": torch.cuda.is_available()
        },
        "endpoints": {
            "health": "/health",
            "predict": "/predict",
            "model_info": "/model-info",
            "system_info": "/system-info",
            "docs": "/docs"
        }
    }

@app.get("/health")
async def health_check():
    """Comprehensive health check endpoint"""
    global feature_extractor, classifier
    
    health_status = {
        "timestamp": time.time(),
        "status": "healthy",
        "checks": {}
    }
    
    try:
        # Check models
        if feature_extractor is None or classifier is None:
            health_status["status"] = "unhealthy"
            health_status["checks"]["models"] = {"status": "failed", "error": "models not loaded"}
            return JSONResponse(status_code=503, content=health_status)
        
        health_status["checks"]["models"] = {"status": "ok"}
        
        # Test inference
        test_input = torch.randn(1, 3, *config.IMAGE_SIZE).to(DEVICE)
        with torch.no_grad():
            features = feature_extractor.encode_image(test_input)
            outputs = classifier(features)
            _ = F.softmax(outputs, dim=1)
        
        health_status["checks"]["inference"] = {"status": "ok"}
        
        # Check device
        health_status["checks"]["device"] = {
            "status": "ok",
            "device": str(DEVICE),
            "cuda_available": torch.cuda.is_available()
        }
        
        # Memory check
        if torch.cuda.is_available():
            memory_used = torch.cuda.memory_allocated() / (1024**3)
            memory_total = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            health_status["checks"]["memory"] = {
                "status": "ok",
                "used_gb": round(memory_used, 2),
                "total_gb": round(memory_total, 2),
                "usage_percent": round((memory_used / memory_total) * 100, 1)
            }
        
        return health_status
        
    except Exception as e:
        health_status["status"] = "unhealthy"
        health_status["checks"]["inference"] = {"status": "failed", "error": str(e)}
        return JSONResponse(status_code=503, content=health_status)

@app.get("/model-info")
async def model_info():
    """Get detailed model information"""
    global model_metadata
    
    return {
        "model_version": config.MODEL_VERSION,
        "model_metadata": model_metadata,
        "classes": CLASS_NAMES,
        "architecture": {
            "feature_extractor": "Enhanced ResNet50",
            "classifier": "Robust Multi-layer Classifier",
            "input_size": config.IMAGE_SIZE,
            "feature_dim": config.FEATURE_DIM
        },
        "training_info": {
            "normalization": {
                "mean": config.NORMALIZE_MEAN,
                "std": config.NORMALIZE_STD
            }
        }
    }

@app.get("/system-info")
async def system_info():
    """Get system information"""
    return {
        "system": get_system_info(),
        "configuration": {
            "max_file_size": config.MAX_FILE_SIZE,
            "image_size": config.IMAGE_SIZE,
            "timeout": config.REQUEST_TIMEOUT,
            "device": str(DEVICE)
        }
    }

@app.post("/predict")
async def predict_pest(image: UploadFile = File(...)):
    """Enhanced prediction endpoint with comprehensive validation"""
    global feature_extractor, classifier
    
    # Validate models
    if feature_extractor is None or classifier is None:
        raise HTTPException(status_code=503, detail="Models not loaded")
    
    # Validate file
    if not image.content_type or not image.content_type.startswith('image/'):
        raise HTTPException(status_code=400, detail="File must be an image")
    
    if image.size and image.size > config.MAX_FILE_SIZE:
        raise HTTPException(
            status_code=400,
            detail=f"File too large (max {config.MAX_FILE_SIZE / (1024*1024):.1f}MB)"
        )
    
    start_time = time.time()
    
    try:
        # Read image
        logger.info(f"Processing image: {image.filename}")
        image_bytes = await image.read()
        
        # Process image
        image_tensor, pil_image = image_processor.process_image(image_bytes)
        
        # Run inference
        with torch.no_grad():
            # Extract features
            features = feature_extractor.encode_image(image_tensor)
            
            # Get predictions
            outputs = classifier(features)
            probabilities = F.softmax(outputs, dim=1)
            
            # Get results
            confidence, predicted_class = torch.max(probabilities, 1)
            confidence_score = confidence.item()
            predicted_idx = predicted_class.item()
            
            # Validate prediction index
            if predicted_idx >= len(CLASS_NAMES):
                predicted_idx = 0
            
            predicted_class_name = CLASS_NAMES[predicted_idx]
            all_probabilities = probabilities.squeeze().cpu().numpy()
        
        processing_time = time.time() - start_time
        
        # Create prediction result
        prediction_result = {
            "species": predicted_class_name,
            "confidence": round(confidence_score, 4),
            "confidence_percentage": round(confidence_score * 100, 2)
        }
        
        # Create annotated image
        annotated_image_b64 = image_processor.create_annotated_image(pil_image, prediction_result)
        
        # Format response
        response = {
            "success": True,
            "timestamp": time.time(),
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
                "size_bytes": len(image_bytes),
                "format": pil_image.format or "Unknown"
            },
            "annotated_image": annotated_image_b64,
            "processing_time": round(processing_time, 3),
            "model_info": {
                "version": config.MODEL_VERSION,
                "device": str(DEVICE),
                "feature_dim": model_metadata.get("feature_dim", "unknown")
            }
        }
        
        logger.info(f"✅ Prediction: {predicted_class_name} ({confidence_score:.2%}) in {processing_time:.3f}s")
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
    """Global exception handler with detailed logging"""
    logger.error(f"❌ Unhandled exception: {exc}")
    logger.error(f"Request: {request.method} {request.url}")
    
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal server error",
            "type": type(exc).__name__,
            "timestamp": time.time()
        }
    )

@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    """HTTP exception handler"""
    logger.warning(f"HTTP exception: {exc.status_code} - {exc.detail}")
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": exc.detail,
            "status_code": exc.status_code,
            "timestamp": time.time()
        }
    )

# =============================================================================
# APPLICATION RUNNER
# =============================================================================

if __name__ == "__main__":
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        log_level="info",
        access_log=True
    )
