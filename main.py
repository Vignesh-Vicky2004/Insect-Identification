import os
import sys
import requests
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import base64
import asyncio
from PIL import Image, ImageDraw, ImageFont
from io import BytesIO
import logging
import time
from contextlib import asynccontextmanager
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import traceback
from typing import Optional, Dict, Any

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
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
    def __init__(self, input_dim: int, num_classes: int, dropout_rate: float = 0.3):
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

def check_dependencies() -> bool:
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
        return False
    return True

def get_open_clip_version() -> str:
    """Get open_clip version with multiple fallback methods"""
    try:
        import open_clip
        
        # Method 1: Direct __version__ attribute
        if hasattr(open_clip, '__version__'):
            return open_clip.__version__
        
        # Method 2: Try pkg_resources
        try:
            import pkg_resources
            return pkg_resources.get_distribution('open_clip_torch').version
        except:
            pass
        
        # Method 3: Try importlib.metadata (Python 3.8+)
        try:
            from importlib import metadata
            return metadata.version('open_clip_torch')
        except:
            pass
        
        # Method 4: Check if it's working
        try:
            # Test basic functionality
            open_clip.list_models()
            return "Available (version unknown)"
        except:
            return "Not functional"
            
    except ImportError:
        return "Not installed"
    except Exception as e:
        return f"Error: {str(e)}"

def get_system_info() -> Dict[str, Any]:
    """Get comprehensive system information"""
    info = {
        "python_version": sys.version,
        "torch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_device_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
        "device": str(DEVICE),
        "open_clip_version": get_open_clip_version()
    }
    
    if torch.cuda.is_available():
        try:
            info["cuda_device_name"] = torch.cuda.get_device_name(0)
            info["cuda_memory_total"] = torch.cuda.get_device_properties(0).total_memory
            info["cuda_memory_allocated"] = torch.cuda.memory_allocated(0)
            info["cuda_memory_cached"] = torch.cuda.memory_reserved(0)
        except Exception as e:
            info["cuda_error"] = str(e)
    
    # Memory information
    try:
        import psutil
        process = psutil.Process(os.getpid())
        memory_info = process.memory_info()
        info["memory_usage"] = {
            "rss_mb": round(memory_info.rss / 1024 / 1024, 2),
            "vms_mb": round(memory_info.vms / 1024 / 1024, 2),
            "percent": round(process.memory_percent(), 2)
        }
    except ImportError:
        info["memory_usage"] = "psutil not available"
    
    return info

async def download_trained_classifier():
    """Download classifier with enhanced error handling and multiple attempts"""
    max_retries = 3
    retry_delay = 5
    
    # Create directory if needed
    os.makedirs(os.path.dirname(LOCAL_MODEL_PATH) if os.path.dirname(LOCAL_MODEL_PATH) else '.', exist_ok=True)
    
    for attempt in range(max_retries):
        try:
            logger.info(f"Downloading classifier (attempt {attempt + 1}/{max_retries})...")
            
            # Enhanced headers for better compatibility
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
                'Accept': 'application/octet-stream, */*',
                'Accept-Encoding': 'identity',
                'Connection': 'keep-alive',
                'Cache-Control': 'no-cache'
            }
            
            # Use session for better connection handling
            session = requests.Session()
            session.headers.update(headers)
            
            # Download with progress tracking
            response = session.get(GITHUB_MODEL_URL, stream=True, timeout=120)
            response.raise_for_status()
            
            total_size = int(response.headers.get('content-length', 0))
            downloaded = 0
            
            logger.info(f"Starting download - Total size: {total_size} bytes")
            
            with open(LOCAL_MODEL_PATH, 'wb') as f:
                for chunk in response.iter_content(chunk_size=16384):
                    if chunk:
                        f.write(chunk)
                        downloaded += len(chunk)
                        
                        # Log progress every 1MB
                        if total_size > 0 and downloaded % (1024 * 1024) == 0:
                            progress = (downloaded / total_size) * 100
                            logger.info(f"Download progress: {progress:.1f}% ({downloaded:,}/{total_size:,} bytes)")
            
            # Verify download
            file_size = os.path.getsize(LOCAL_MODEL_PATH)
            logger.info(f"Download completed - File size: {file_size:,} bytes")
            
            if file_size < 1000000:  # Less than 1MB is suspicious
                raise ValueError(f"Downloaded file too small: {file_size} bytes")
            
            # Test file integrity
            try:
                # Try to load the model file
                test_load = torch.load(LOCAL_MODEL_PATH, map_location='cpu')
                logger.info("✓ Model file integrity verified")
                return
                
            except Exception as e:
                raise ValueError(f"Downloaded model file is corrupted: {e}")
                
        except Exception as e:
            logger.error(f"Download attempt {attempt + 1} failed: {e}")
            
            # Clean up corrupted file
            if os.path.exists(LOCAL_MODEL_PATH):
                os.remove(LOCAL_MODEL_PATH)
            
            if attempt < max_retries - 1:
                logger.info(f"Retrying in {retry_delay} seconds...")
                await asyncio.sleep(retry_delay)
                retry_delay *= 2
            else:
                raise RuntimeError(f"Failed to download classifier after {max_retries} attempts: {e}")

async def load_bioclip_models():
    """Load BioCLIP models with comprehensive error handling"""
    global bioclip_model, bioclip_preprocess, classifier
    
    try:
        # Check dependencies first
        if not check_dependencies():
            raise RuntimeError("Missing required dependencies")
        
        # Set up model caching environment
        cache_dirs = ['/app/models', './model_cache', './cache']
        for cache_dir in cache_dirs:
            try:
                os.makedirs(cache_dir, exist_ok=True)
                os.environ['HF_HOME'] = cache_dir
                os.environ['TRANSFORMERS_CACHE'] = cache_dir
                os.environ['HF_HUB_CACHE'] = cache_dir
                os.environ['TORCH_HOME'] = cache_dir
                break
            except:
                continue
        
        # Import open_clip with error handling
        try:
            import open_clip
            logger.info("✓ open_clip imported successfully")
            logger.info(f"open_clip version: {get_open_clip_version()}")
        except Exception as e:
            logger.error(f"Failed to import open_clip: {e}")
            raise RuntimeError(f"open_clip import failed: {e}")
        
        logger.info("Loading BioCLIP model...")
        
        # Model loading with multiple fallback methods
        model_name = 'hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224'
        loaded = False
        
        # Method 1: Standard loading
        try:
            logger.info("Attempting standard model loading...")
            bioclip_model, _, bioclip_preprocess = open_clip.create_model_and_transforms(
                model_name,
                cache_dir=os.environ.get('HF_HOME', './model_cache')
            )
            loaded = True
            logger.info("✓ BioCLIP loaded with standard method")
        except Exception as e1:
            logger.warning(f"Standard loading failed: {e1}")
            
            # Method 2: With explicit device
            try:
                logger.info("Attempting loading with explicit device...")
                bioclip_model, _, bioclip_preprocess = open_clip.create_model_and_transforms(
                    model_name,
                    device=DEVICE,
                    cache_dir=os.environ.get('HF_HOME', './model_cache')
                )
                loaded = True
                logger.info("✓ BioCLIP loaded with device specification")
            except Exception as e2:
                logger.warning(f"Device-specific loading failed: {e2}")
                
                # Method 3: Force download
                try:
                    logger.info("Attempting force download...")
                    bioclip_model, _, bioclip_preprocess = open_clip.create_model_and_transforms(
                        model_name,
                        force_quick_gelu=False,
                        cache_dir=os.environ.get('HF_HOME', './model_cache')
                    )
                    loaded = True
                    logger.info("✓ BioCLIP loaded with force download")
                except Exception as e3:
                    logger.error(f"All BioCLIP loading methods failed: {e1}, {e2}, {e3}")
                    raise RuntimeError(f"Failed to load BioCLIP model: {e3}")
        
        if not loaded:
            raise RuntimeError("BioCLIP model loading failed")
        
        # Move model to device and set evaluation mode
        bioclip_model = bioclip_model.to(DEVICE)
        bioclip_model.eval()
        
        # Optimize model for inference if possible
        try:
            if hasattr(torch.jit, 'optimize_for_inference'):
                bioclip_model = torch.jit.optimize_for_inference(bioclip_model)
                logger.info("✓ Model optimized for inference")
        except:
            logger.info("Model optimization not available")
        
        logger.info(f"BioCLIP model loaded and moved to device: {DEVICE}")
        
        # Test model and get feature dimension
        try:
            dummy_input = torch.randn(1, 3, 224, 224).to(DEVICE)
            with torch.no_grad():
                features = bioclip_model.encode_image(dummy_input)
                feature_dim = features.shape[-1]
            logger.info(f"✓ BioCLIP test successful - Feature dimension: {feature_dim}")
        except Exception as e:
            logger.error(f"BioCLIP model test failed: {e}")
            raise RuntimeError(f"BioCLIP model test failed: {e}")
        
        # Load classifier
        await load_classifier(feature_dim)
        
        # Final pipeline test
        try:
            with torch.no_grad():
                test_features = bioclip_model.encode_image(dummy_input)
                test_output = classifier(test_features)
                test_probs = F.softmax(test_output, dim=1)
                
                if test_probs.shape[1] != len(CLASS_NAMES):
                    raise ValueError(f"Output dimension mismatch: {test_probs.shape[1]} vs {len(CLASS_NAMES)}")
                
                logger.info("✓ Complete pipeline test successful")
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

async def load_classifier(feature_dim: int):
    """Load classifier with enhanced error handling"""
    global classifier
    
    # Download classifier if needed
    if not os.path.exists(LOCAL_MODEL_PATH):
        logger.info("Classifier not found locally, downloading...")
        await download_trained_classifier()
    else:
        logger.info(f"Using local classifier file: {LOCAL_MODEL_PATH}")
        logger.info(f"File size: {os.path.getsize(LOCAL_MODEL_PATH):,} bytes")
    
    # Load classifier
    try:
        classifier = OptimizedClassifier(feature_dim, len(CLASS_NAMES))
        logger.info(f"Created classifier with input_dim={feature_dim}, num_classes={len(CLASS_NAMES)}")
        
        # Load state dict with enhanced error handling
        try:
            # Try with weights_only=True for security (PyTorch >= 1.13)
            state_dict = torch.load(LOCAL_MODEL_PATH, map_location=DEVICE, weights_only=True)
            logger.info("✓ Model loaded with weights_only=True")
        except TypeError:
            # Fallback for older PyTorch versions
            state_dict = torch.load(LOCAL_MODEL_PATH, map_location=DEVICE)
            logger.info("✓ Model loaded with fallback method")
        
        # Check state dict structure
        model_keys = set(classifier.state_dict().keys())
        loaded_keys = set(state_dict.keys())
        
        if model_keys == loaded_keys:
            classifier.load_state_dict(state_dict, strict=True)
            logger.info("✓ Classifier loaded with strict=True")
        else:
            logger.warning("State dict keys mismatch:")
            logger.warning(f"Model keys: {len(model_keys)}, Loaded keys: {len(loaded_keys)}")
            
            missing_keys = model_keys - loaded_keys
            unexpected_keys = loaded_keys - model_keys
            
            if missing_keys:
                logger.warning(f"Missing keys: {missing_keys}")
            if unexpected_keys:
                logger.warning(f"Unexpected keys: {unexpected_keys}")
            
            # Try loading with strict=False
            result = classifier.load_state_dict(state_dict, strict=False)
            if result.missing_keys or result.unexpected_keys:
                logger.warning(f"Load result - Missing: {result.missing_keys}, Unexpected: {result.unexpected_keys}")
            
            logger.info("✓ Classifier loaded with strict=False")
        
        # Move to device and set evaluation mode
        classifier = classifier.to(DEVICE)
        classifier.eval()
        
        # Test classifier
        test_features = torch.randn(1, feature_dim).to(DEVICE)
        with torch.no_grad():
            test_output = classifier(test_features)
            test_probs = F.softmax(test_output, dim=1)
            
            if test_output.shape[1] != len(CLASS_NAMES):
                raise ValueError(f"Classifier output dimension mismatch: {test_output.shape[1]} vs {len(CLASS_NAMES)}")
            
            logger.info(f"✓ Classifier test successful - Output shape: {test_output.shape}")
        
    except Exception as e:
        logger.error(f"Classifier loading failed: {e}")
        raise RuntimeError(f"Classifier loading failed: {e}")

def preprocess_image(pil_image: Image.Image) -> torch.Tensor:
    """Preprocess image with fallback options"""
    global bioclip_preprocess
    
    if bioclip_preprocess is None:
        logger.warning("BioCLIP preprocess not available, using fallback")
        import torchvision.transforms as transforms
        fallback_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
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

def create_prediction_visualization(image: Image.Image, prediction: Dict[str, Any]) -> str:
    """Create visualization with comprehensive error handling"""
    try:
        # Resize very large images
        max_size = 1024
        if max(image.size) > max_size:
            image.thumbnail((max_size, max_size), Image.Resampling.LANCZOS)
        
        # Convert to RGB if needed
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        annotated = image.copy()
        draw = ImageDraw.Draw(annotated)
        
        # Try to load a nice font
        font = None
        font_paths = [
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
            "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
            "/System/Library/Fonts/Helvetica.ttc",
            "/System/Library/Fonts/Arial.ttf",
            "C:/Windows/Fonts/arial.ttf",
            "C:/Windows/Fonts/calibri.ttf"
        ]
        
        for font_path in font_paths:
            try:
                font = ImageFont.truetype(font_path, 24)
                break
            except (OSError, IOError):
                continue
        
        # Fallback to default font
        if font is None:
            try:
                font = ImageFont.load_default()
            except:
                font = None
        
        # Prepare text
        species = prediction['species']
        confidence = prediction['confidence_percentage']
        main_text = f"Prediction: {species}"
        conf_text = f"Confidence: {confidence:.1f}%"
        
        # Color coding based on confidence
        if confidence >= 85:
            bg_color, text_color = "#2ECC71", "white"  # Green
        elif confidence >= 70:
            bg_color, text_color = "#F39C12", "white"  # Orange
        elif confidence >= 50:
            bg_color, text_color = "#E67E22", "white"  # Dark Orange
        else:
            bg_color, text_color = "#E74C3C", "white"  # Red
        
        # Calculate text dimensions
        if font:
            main_bbox = draw.textbbox((0, 0), main_text, font=font)
            conf_bbox = draw.textbbox((0, 0), conf_text, font=font)
        else:
            # Estimate dimensions if font is None
            main_bbox = (0, 0, len(main_text) * 8, 16)
            conf_bbox = (0, 0, len(conf_text) * 8, 16)
        
        # Position and dimensions
        x, y = 15, 15
        padding = 12
        
        width = max(main_bbox[2] - main_bbox[0], conf_bbox[2] - conf_bbox[0]) + 2 * padding
        height = (main_bbox[3] - main_bbox[1]) + (conf_bbox[3] - conf_bbox[1]) + 3 * padding
        
        # Draw background rectangle
        draw.rectangle([x, y, x + width, y + height], fill=bg_color, outline="#34495E", width=3)
        
        # Draw text
        if font:
            draw.text((x + padding, y + padding), main_text, fill=text_color, font=font)
            draw.text((x + padding, y + padding + (main_bbox[3] - main_bbox[1]) + padding), 
                     conf_text, fill=text_color, font=font)
        else:
            # Fallback without font
            draw.text((x + padding, y + padding), main_text, fill=text_color)
            draw.text((x + padding, y + padding + 20), conf_text, fill=text_color)
        
        # Convert to base64
        buffer = BytesIO()
        annotated.save(buffer, format='JPEG', quality=95, optimize=True)
        return base64.b64encode(buffer.getvalue()).decode('utf-8')
        
    except Exception as e:
        logger.error(f"Visualization creation failed: {e}")
        # Return original image as fallback
        try:
            buffer = BytesIO()
            if image.mode != 'RGB':
                image = image.convert('RGB')
            image.save(buffer, format='JPEG', quality=95)
            return base64.b64encode(buffer.getvalue()).decode('utf-8')
        except Exception as e2:
            logger.error(f"Fallback visualization also failed: {e2}")
            raise HTTPException(status_code=500, detail="Failed to create visualization")

# FastAPI Application Setup
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Enhanced application lifespan management"""
    logger.info("=" * 60)
    logger.info("🚀 Starting BioCLIP Pest Identification API")
    logger.info("=" * 60)
    
    # Log system information
    system_info = get_system_info()
    logger.info(f"System Information:")
    for key, value in system_info.items():
        logger.info(f"  {key}: {value}")
    
    startup_success = False
    startup_error = None
    
    try:
        start_time = time.time()
        await load_bioclip_models()
        load_time = time.time() - start_time
        
        startup_success = True
        logger.info("=" * 60)
        logger.info(f"✅ BioCLIP API Server ready! (Load time: {load_time:.2f}s)")
        logger.info("=" * 60)
        
    except Exception as e:
        startup_error = str(e)
        logger.error("=" * 60)
        logger.error(f"❌ Startup failed: {e}")
        logger.error("🔧 Server will start but models are not available")
        logger.error("🔧 Use /reload endpoint to retry model loading")
        logger.error("=" * 60)
    
    # Store startup info in app state
    app.state.startup_success = startup_success
    app.state.startup_error = startup_error
    app.state.startup_time = time.time()
    
    yield
    
    logger.info("🛑 Shutting down BioCLIP API Server...")

# Initialize FastAPI app
app = FastAPI(
    title="BioCLIP Pest Identification API",
    version="3.2.2",
    description="Advanced pest identification API using BioCLIP foundation model with comprehensive error handling and monitoring",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc"
)

# Add CORS middleware
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
    """Root endpoint with comprehensive API information"""
    global bioclip_model, classifier
    
    model_status = "loaded" if bioclip_model is not None and classifier is not None else "not_loaded"
    
    return {
        "message": "BioCLIP Pest Identification API",
        "version": "3.2.2",
        "status": "running",
        "model_status": model_status,
        "startup_success": getattr(app.state, 'startup_success', False),
        "startup_error": getattr(app.state, 'startup_error', None),
        "uptime": time.time() - getattr(app.state, 'startup_time', time.time()),
        "endpoints": {
            "health": "/health - Check API health and model status",
            "predict": "/predict - Upload image for pest identification",
            "debug": "/debug - System and model debug information",
            "reload": "/reload - Manually reload models",
            "docs": "/docs - API documentation"
        },
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
    """Comprehensive health check endpoint"""
    global bioclip_model, classifier, bioclip_preprocess
    
    check_time = time.time()
    
    # Basic model availability check
    models_available = {
        "bioclip_model": bioclip_model is not None,
        "classifier": classifier is not None,
        "preprocess": bioclip_preprocess is not None
    }
    
    health_data = {
        "timestamp": check_time,
        "startup_success": getattr(app.state, 'startup_success', False),
        "startup_error": getattr(app.state, 'startup_error', None),
        "uptime": check_time - getattr(app.state, 'startup_time', check_time),
        "system_info": get_system_info(),
        "model_status": models_available,
        "file_status": {
            "classifier_exists": os.path.exists(LOCAL_MODEL_PATH),
            "classifier_size": os.path.getsize(LOCAL_MODEL_PATH) if os.path.exists(LOCAL_MODEL_PATH) else 0
        }
    }
    
    if not all(models_available.values()):
        return JSONResponse(
            status_code=503,
            content={
                "healthy": False,
                "status": "models_not_loaded",
                "message": "One or more models are not loaded. Use /reload to retry.",
                **health_data
            }
        )
    
    # Perform model functionality test
    try:
        test_input = torch.randn(1, 3, 224, 224).to(DEVICE)
        
        with torch.no_grad():
            start_time = time.time()
            
            # Test BioCLIP feature extraction
            features = bioclip_model.encode_image(test_input)
            
            # Test classifier
            outputs = classifier(features)
            probabilities = F.softmax(outputs, dim=1)
            
            inference_time = time.time() - start_time
            
        return {
            "healthy": True,
            "status": "fully_operational",
            "performance": {
                "inference_time": round(inference_time, 4),
                "feature_dimension": features.shape[-1],
                "output_classes": probabilities.shape[-1],
                "throughput_estimate": round(1.0 / inference_time, 2)
            },
            **health_data
        }
        
    except Exception as e:
        return JSONResponse(
            status_code=503,
            content={
                "healthy": False,
                "status": "model_test_failed",
                "error": str(e),
                "message": "Models are loaded but functionality test failed",
                **health_data
            }
        )

@app.post("/predict")
async def predict_pest(image: UploadFile = File(...)):
    """Enhanced pest prediction endpoint with comprehensive validation"""
    global bioclip_model, classifier, bioclip_preprocess
    
    # Validate models are loaded
    if any(model is None for model in [bioclip_model, classifier]):
        raise HTTPException(
            status_code=503, 
            detail="BioCLIP models not loaded. Check /health endpoint for details."
        )
    
    # Validate file type
    if not image.content_type or not image.content_type.startswith('image/'):
        raise HTTPException(
            status_code=400, 
            detail=f"Invalid file type. Expected image, got: {image.content_type}"
        )
    
    # Validate filename
    if not image.filename:
        raise HTTPException(status_code=400, detail="No filename provided")
    
    try:
        # Read and validate image data
        image_bytes = await image.read()
        
        if len(image_bytes) == 0:
            raise HTTPException(status_code=400, detail="Empty image file")
        
        if len(image_bytes) > 15 * 1024 * 1024:  # 15MB limit
            raise HTTPException(status_code=400, detail="Image too large (max 15MB)")
        
        # Process image
        try:
            pil_image = Image.open(BytesIO(image_bytes))
            
            # Validate image
            if pil_image.size[0] < 32 or pil_image.size[1] < 32:
                raise HTTPException(status_code=400, detail="Image too small (minimum 32x32 pixels)")
            
            if pil_image.size[0] > 4096 or pil_image.size[1] > 4096:
                raise HTTPException(status_code=400, detail="Image too large (maximum 4096x4096 pixels)")
            
            # Convert to RGB if needed
            if pil_image.mode != 'RGB':
                pil_image = pil_image.convert('RGB')
                
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Invalid image format: {str(e)}")
        
        # Preprocess image
        try:
            image_tensor = preprocess_image(pil_image).to(DEVICE)
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Image preprocessing failed: {str(e)}")
        
        # Make prediction
        start_time = time.time()
        
        try:
            with torch.no_grad():
                # Extract features
                features = bioclip_model.encode_image(image_tensor)
                
                # Classify
                outputs = classifier(features)
                probabilities = F.softmax(outputs, dim=1)
                
                # Get prediction
                confidence, predicted_class = torch.max(probabilities, 1)
                predicted_idx = predicted_class.item()
                predicted_species = CLASS_NAMES[predicted_idx]
                
                # Get all class scores
                all_probs = probabilities.cpu().numpy().flatten()
                class_scores = {
                    class_name: round(float(prob), 4) 
                    for class_name, prob in zip(CLASS_NAMES, all_probs)
                }
                
                # Sort class scores by confidence
                sorted_scores = dict(sorted(class_scores.items(), key=lambda x: x[1], reverse=True))
                
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")
        
        processing_time = time.time() - start_time
        
        # Create prediction result
        prediction_result = {
            "species": predicted_species,
            "confidence": round(confidence.item(), 4),
            "confidence_percentage": round(confidence.item() * 100, 2),
            "all_class_scores": sorted_scores,
            "prediction_quality": "high" if confidence.item() > 0.8 else "medium" if confidence.item() > 0.6 else "low"
        }
        
        # Create visualization
        try:
            annotated_image = create_prediction_visualization(pil_image, prediction_result)
        except Exception as e:
            logger.error(f"Visualization failed: {e}")
            annotated_image = None
        
        # Prepare response
        response = {
            "success": True,
            "prediction": prediction_result,
            "metadata": {
                "processing_time": round(processing_time, 4),
                "device": str(DEVICE),
                "image_size": pil_image.size,
                "image_format": pil_image.format,
                "file_size": len(image_bytes),
                "timestamp": time.time()
            }
        }
        
        # Add visualization if available
        if annotated_image:
            response["annotated_image"] = annotated_image
        
        return response
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Unexpected prediction error: {e}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

@app.get("/debug")
async def debug_info():
    """Comprehensive debug information endpoint"""
    global bioclip_model, classifier, bioclip_preprocess
    
    try:
        debug_data = {
            "timestamp": time.time(),
            "dependencies": check_dependencies(),
            "system_info": get_system_info(),
            "model_status": {
                "bioclip_model": bioclip_model is not None,
                "classifier": classifier is not None,
                "preprocess": bioclip_preprocess is not None
            },
            "file_status": {
                "classifier_exists": os.path.exists(LOCAL_MODEL_PATH),
                "classifier_size": os.path.getsize(LOCAL_MODEL_PATH) if os.path.exists(LOCAL_MODEL_PATH) else 0,
                "classifier_path": LOCAL_MODEL_PATH
            },
            "device_info": {
                "device": str(DEVICE),
                "cuda_available": torch.cuda.is_available(),
                "cuda_device_count": torch.cuda.device_count() if torch.cuda.is_available() else 0
            },
            "configuration": {
                "class_names": CLASS_NAMES,
                "num_classes": len(CLASS_NAMES),
                "model_url": GITHUB_MODEL_URL
            },
            "startup_info": {
                "startup_success": getattr(app.state, 'startup_success', False),
                "startup_error": getattr(app.state, 'startup_error', None),
                "uptime": time.time() - getattr(app.state, 'startup_time', time.time())
            }
        }
        
        # Add model details if available
        if bioclip_model is not None:
            try:
                # Test model to get feature dimension
                test_input = torch.randn(1, 3, 224, 224).to(DEVICE)
                with torch.no_grad():
                    features = bioclip_model.encode_image(test_input)
                    debug_data["model_details"] = {
                        "feature_dimension": features.shape[-1],
                        "input_shape": test_input.shape,
                        "feature_shape": features.shape
                    }
            except Exception as e:
                debug_data["model_details"] = {"error": str(e)}
        
        return debug_data
        
    except Exception as e:
        logger.error(f"Debug info generation failed: {e}")
        return {
            "error": str(e),
            "timestamp": time.time(),
            "basic_info": {
                "python_version": sys.version,
                "torch_version": torch.__version__
            }
        }

@app.post("/reload")
async def reload_models():
    """Enhanced model reload endpoint"""
    try:
        logger.info("🔄 Manual model reload requested...")
        
        # Reset global variables
        global bioclip_model, classifier, bioclip_preprocess
        bioclip_model = None
        classifier = None
        bioclip_preprocess = None
        
        # Clear GPU cache if available
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            logger.info("GPU cache cleared")
        
        # Reload models
        start_time = time.time()
        await load_bioclip_models()
        load_time = time.time() - start_time
        
        # Update app state
        app.state.startup_success = True
        app.state.startup_error = None
        app.state.startup_time = time.time()
        
        logger.info(f"✅ Models reloaded successfully in {load_time:.2f}s")
        
        return {
            "success": True,
            "message": "Models reloaded successfully",
            "load_time": round(load_time, 2),
            "timestamp": time.time()
        }
        
    except Exception as e:
        logger.error(f"❌ Manual reload failed: {e}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        
        # Update app state
        app.state.startup_success = False
        app.state.startup_error = str(e)
        
        return JSONResponse(
            status_code=500,
            content={
                "success": False,
                "error": str(e),
                "message": "Model reload failed",
                "timestamp": time.time()
            }
        )

# Error handlers
@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    """Custom HTTP exception handler"""
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": exc.detail,
            "status_code": exc.status_code,
            "timestamp": time.time()
        }
    )

@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    """General exception handler"""
    logger.error(f"Unhandled exception: {exc}")
    logger.error(f"Traceback: {traceback.format_exc()}")
    
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal server error",
            "message": str(exc),
            "timestamp": time.time()
        }
    )

# Main execution
if __name__ == "__main__":
    import uvicorn
    
    # Configure uvicorn logging
    logging.getLogger("uvicorn.access").setLevel(logging.INFO)
    
    uvicorn.run(
        app, 
        host="0.0.0.0", 
        port=8000, 
        reload=False,
        workers=1,
        timeout_keep_alive=30,
        access_log=True
    )
