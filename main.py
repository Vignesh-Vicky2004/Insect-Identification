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
from typing import Optional, Dict, Any, List
import json
from pathlib import Path

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
model_loading_lock = asyncio.Lock()

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

# Enhanced Model Architecture
class OptimizedClassifier(nn.Module):
    def __init__(self, input_dim: int, num_classes: int, dropout_rate: float = 0.3):
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
        'requests': 'requests',
        'huggingface_hub': 'huggingface_hub'
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

def setup_model_cache():
    """Setup model cache directories with proper permissions"""
    cache_dirs = [
        '/app/models',
        './model_cache', 
        './cache',
        os.path.expanduser('~/.cache/huggingface'),
        os.path.expanduser('~/.cache/torch')
    ]
    
    for cache_dir in cache_dirs:
        try:
            os.makedirs(cache_dir, exist_ok=True)
            # Set environment variables
            os.environ['HF_HOME'] = cache_dir
            os.environ['TRANSFORMERS_CACHE'] = cache_dir
            os.environ['HF_HUB_CACHE'] = cache_dir
            os.environ['TORCH_HOME'] = cache_dir
            logger.info(f"✓ Cache directory set up: {cache_dir}")
            return cache_dir
        except Exception as e:
            logger.warning(f"Failed to setup cache dir {cache_dir}: {e}")
            continue
    
    logger.warning("No cache directory could be set up")
    return None

async def download_with_retry(url: str, filepath: str, max_retries: int = 5) -> bool:
    """Download file with retry logic and better error handling"""
    
    for attempt in range(max_retries):
        try:
            logger.info(f"Downloading {url} (attempt {attempt + 1}/{max_retries})")
            
            # Enhanced headers
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
                'Accept': 'application/octet-stream, */*',
                'Accept-Encoding': 'identity',
                'Connection': 'keep-alive',
                'Cache-Control': 'no-cache'
            }
            
            # Configure session with timeouts
            session = requests.Session()
            session.headers.update(headers)
            
            # Download with streaming
            response = session.get(url, stream=True, timeout=(30, 300))
            response.raise_for_status()
            
            total_size = int(response.headers.get('content-length', 0))
            downloaded = 0
            
            logger.info(f"Starting download - Total size: {total_size:,} bytes")
            
            with open(filepath, 'wb') as f:
                for chunk in response.iter_content(chunk_size=32768):
                    if chunk:
                        f.write(chunk)
                        downloaded += len(chunk)
                        
                        # Log progress every 5MB
                        if total_size > 0 and downloaded % (5 * 1024 * 1024) == 0:
                            progress = (downloaded / total_size) * 100
                            logger.info(f"Progress: {progress:.1f}% ({downloaded:,}/{total_size:,} bytes)")
            
            # Verify download
            file_size = os.path.getsize(filepath)
            
            if file_size < 100000:  # Less than 100KB is suspicious
                raise ValueError(f"Downloaded file too small: {file_size} bytes")
            
            # Test file integrity for PyTorch models
            if filepath.endswith('.pth'):
                try:
                    torch.load(filepath, map_location='cpu')
                    logger.info("✓ PyTorch model file integrity verified")
                except Exception as e:
                    raise ValueError(f"Downloaded model file corrupted: {e}")
            
            logger.info(f"✓ Download successful: {file_size:,} bytes")
            return True
            
        except Exception as e:
            logger.error(f"Download attempt {attempt + 1} failed: {e}")
            
            # Clean up corrupted file
            if os.path.exists(filepath):
                try:
                    os.remove(filepath)
                except:
                    pass
            
            if attempt < max_retries - 1:
                wait_time = min(2 ** attempt, 60)  # Exponential backoff, max 60s
                logger.info(f"Retrying in {wait_time} seconds...")
                await asyncio.sleep(wait_time)
            else:
                logger.error(f"All {max_retries} download attempts failed")
                return False
    
    return False

async def load_bioclip_model_with_fallbacks():
    """Load BioCLIP model with comprehensive fallback strategies"""
    global bioclip_model, bioclip_preprocess
    
    # Import open_clip
    try:
        import open_clip
        logger.info("✓ open_clip imported successfully")
    except ImportError as e:
        raise RuntimeError(f"open_clip not available: {e}")
    
    model_name = 'hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224'
    
    # Strategy 1: Try loading from cache first
    cache_dir = setup_model_cache()
    
    strategies = [
        # Local cache loading
        {
            'name': 'Local Cache',
            'args': {'cache_dir': cache_dir} if cache_dir else {}
        },
        # Standard loading
        {
            'name': 'Standard Loading',
            'args': {}
        },
        # Force download
        {
            'name': 'Force Download',
            'args': {'force_download': True, 'cache_dir': cache_dir} if cache_dir else {'force_download': True}
        },
        # Retry with different settings
        {
            'name': 'Alternative Loading',
            'args': {'pretrained': 'openai', 'cache_dir': cache_dir} if cache_dir else {'pretrained': 'openai'}
        }
    ]
    
    for i, strategy in enumerate(strategies):
        try:
            logger.info(f"Attempting {strategy['name']} (strategy {i+1}/{len(strategies)})")
            
            # Add timeout handling
            model, _, preprocess = await asyncio.wait_for(
                asyncio.to_thread(
                    open_clip.create_model_and_transforms,
                    model_name,
                    **strategy['args']
                ),
                timeout=300  # 5 minutes timeout
            )
            
            # Test model functionality
            dummy_input = torch.randn(1, 3, 224, 224)
            with torch.no_grad():
                features = model.encode_image(dummy_input)
                if features.shape[-1] == 0:
                    raise ValueError("Model returned empty features")
            
            bioclip_model = model.to(DEVICE)
            bioclip_preprocess = preprocess
            
            logger.info(f"✅ {strategy['name']} successful!")
            return True
            
        except asyncio.TimeoutError:
            logger.error(f"{strategy['name']} timed out after 5 minutes")
        except Exception as e:
            logger.warning(f"{strategy['name']} failed: {e}")
            continue
    
    raise RuntimeError("All BioCLIP loading strategies failed")

async def load_classifier_model():
    """Load classifier with enhanced error handling"""
    global classifier
    
    # Download classifier if needed
    if not os.path.exists(LOCAL_MODEL_PATH):
        logger.info("Classifier not found locally, downloading...")
        success = await download_with_retry(GITHUB_MODEL_URL, LOCAL_MODEL_PATH)
        if not success:
            raise RuntimeError("Failed to download classifier model")
    else:
        logger.info(f"Using local classifier: {LOCAL_MODEL_PATH}")
    
    # Get feature dimension from BioCLIP
    try:
        dummy_input = torch.randn(1, 3, 224, 224).to(DEVICE)
        with torch.no_grad():
            features = bioclip_model.encode_image(dummy_input)
            feature_dim = features.shape[-1]
        logger.info(f"Feature dimension: {feature_dim}")
    except Exception as e:
        logger.error(f"Failed to get feature dimension: {e}")
        feature_dim = 768  # Default BioCLIP dimension
    
    # Create classifier
    classifier = OptimizedClassifier(feature_dim, len(CLASS_NAMES))
    
    # Load weights
    try:
        # Try with weights_only for security
        try:
            state_dict = torch.load(LOCAL_MODEL_PATH, map_location=DEVICE, weights_only=True)
        except TypeError:
            # Fallback for older PyTorch versions
            state_dict = torch.load(LOCAL_MODEL_PATH, map_location=DEVICE)
        
        # Load state dict
        missing_keys, unexpected_keys = classifier.load_state_dict(state_dict, strict=False)
        
        if missing_keys:
            logger.warning(f"Missing keys in classifier: {missing_keys}")
        if unexpected_keys:
            logger.warning(f"Unexpected keys in classifier: {unexpected_keys}")
        
        classifier = classifier.to(DEVICE)
        classifier.eval()
        
        # Test classifier
        with torch.no_grad():
            test_output = classifier(torch.randn(1, feature_dim).to(DEVICE))
            if test_output.shape[1] != len(CLASS_NAMES):
                raise ValueError(f"Classifier output mismatch: {test_output.shape[1]} vs {len(CLASS_NAMES)}")
        
        logger.info("✅ Classifier loaded and tested successfully")
        return True
        
    except Exception as e:
        logger.error(f"Classifier loading failed: {e}")
        raise RuntimeError(f"Classifier loading failed: {e}")

async def load_all_models():
    """Load all models with comprehensive error handling"""
    global bioclip_model, bioclip_preprocess, classifier
    
    async with model_loading_lock:
        try:
            # Reset models
            bioclip_model = None
            bioclip_preprocess = None
            classifier = None
            
            # Clear GPU memory
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            # Check dependencies
            if not check_dependencies():
                raise RuntimeError("Missing required dependencies")
            
            # Load BioCLIP model
            logger.info("Loading BioCLIP model...")
            await load_bioclip_model_with_fallbacks()
            
            # Load classifier
            logger.info("Loading classifier...")
            await load_classifier_model()
            
            # Final pipeline test
            logger.info("Testing complete pipeline...")
            dummy_input = torch.randn(1, 3, 224, 224).to(DEVICE)
            with torch.no_grad():
                features = bioclip_model.encode_image(dummy_input)
                outputs = classifier(features)
                probabilities = F.softmax(outputs, dim=1)
                
                if probabilities.shape[1] != len(CLASS_NAMES):
                    raise ValueError(f"Pipeline output mismatch: {probabilities.shape[1]} vs {len(CLASS_NAMES)}")
            
            logger.info("🎉 All models loaded and tested successfully!")
            return True
            
        except Exception as e:
            logger.error(f"Model loading failed: {e}")
            # Reset on failure
            bioclip_model = None
            bioclip_preprocess = None
            classifier = None
            raise

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
        import torchvision.transforms as transforms
        fallback_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        return fallback_transform(pil_image).unsqueeze(0)

def create_prediction_visualization(image: Image.Image, prediction: Dict[str, Any]) -> str:
    """Create enhanced prediction visualization"""
    try:
        # Resize if too large
        max_size = 1024
        if max(image.size) > max_size:
            image.thumbnail((max_size, max_size), Image.Resampling.LANCZOS)
        
        # Ensure RGB mode
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        annotated = image.copy()
        draw = ImageDraw.Draw(annotated)
        
        # Font loading with fallbacks
        font = None
        font_paths = [
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
            "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
            "/System/Library/Fonts/Helvetica.ttc",
            "/System/Library/Fonts/Arial.ttf",
            "C:/Windows/Fonts/arial.ttf"
        ]
        
        for font_path in font_paths:
            try:
                font = ImageFont.truetype(font_path, 24)
                break
            except:
                continue
        
        if font is None:
            font = ImageFont.load_default()
        
        # Prepare text
        species = prediction['species']
        confidence = prediction['confidence_percentage']
        main_text = f"Species: {species}"
        conf_text = f"Confidence: {confidence:.1f}%"
        
        # Dynamic color based on confidence
        if confidence >= 85:
            bg_color, text_color = "#27AE60", "white"  # High confidence - Green
        elif confidence >= 70:
            bg_color, text_color = "#F39C12", "white"  # Medium confidence - Orange
        elif confidence >= 50:
            bg_color, text_color = "#E67E22", "white"  # Low confidence - Dark Orange
        else:
            bg_color, text_color = "#E74C3C", "white"  # Very low confidence - Red
        
        # Calculate text dimensions
        try:
            main_bbox = draw.textbbox((0, 0), main_text, font=font)
            conf_bbox = draw.textbbox((0, 0), conf_text, font=font)
        except:
            # Fallback for older Pillow versions
            main_bbox = (0, 0, len(main_text) * 12, 20)
            conf_bbox = (0, 0, len(conf_text) * 12, 20)
        
        # Position and draw
        x, y = 15, 15
        padding = 12
        
        width = max(main_bbox[2] - main_bbox[0], conf_bbox[2] - conf_bbox[0]) + 2 * padding
        height = (main_bbox[3] - main_bbox[1]) + (conf_bbox[3] - conf_bbox[1]) + 3 * padding
        
        # Draw background with border
        draw.rectangle([x, y, x + width, y + height], fill=bg_color, outline="#2C3E50", width=3)
        
        # Draw text
        draw.text((x + padding, y + padding), main_text, fill=text_color, font=font)
        draw.text((x + padding, y + padding + (main_bbox[3] - main_bbox[1]) + padding), 
                 conf_text, fill=text_color, font=font)
        
        # Convert to base64
        buffer = BytesIO()
        annotated.save(buffer, format='JPEG', quality=95, optimize=True)
        return base64.b64encode(buffer.getvalue()).decode('utf-8')
        
    except Exception as e:
        logger.error(f"Visualization failed: {e}")
        # Return original image as fallback
        buffer = BytesIO()
        if image.mode != 'RGB':
            image = image.convert('RGB')
        image.save(buffer, format='JPEG', quality=95)
        return base64.b64encode(buffer.getvalue()).decode('utf-8')

# FastAPI Application
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Enhanced application lifespan with better error handling"""
    logger.info("🚀 Starting BioCLIP Pest Identification API v4.0")
    logger.info("=" * 60)
    
    # Log system info
    system_info = get_system_info()
    for key, value in system_info.items():
        logger.info(f"  {key}: {value}")
    
    startup_success = False
    startup_error = None
    
    try:
        start_time = time.time()
        await load_all_models()
        load_time = time.time() - start_time
        
        startup_success = True
        logger.info("=" * 60)
        logger.info(f"✅ BioCLIP API ready! Load time: {load_time:.2f}s")
        logger.info("=" * 60)
        
    except Exception as e:
        startup_error = str(e)
        logger.error("=" * 60)
        logger.error(f"❌ Startup failed: {e}")
        logger.error("Server will start but models unavailable")
        logger.error("Use /reload to retry model loading")
        logger.error("=" * 60)
    
    # Store startup state
    app.state.startup_success = startup_success
    app.state.startup_error = startup_error
    app.state.startup_time = time.time()
    
    yield
    
    logger.info("🛑 Shutting down BioCLIP API...")

# Initialize FastAPI
app = FastAPI(
    title="BioCLIP Pest Identification API",
    version="4.0.0",
    description="Production-ready pest identification API using BioCLIP with comprehensive error handling",
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

# API Routes
@app.get("/")
async def root():
    """Enhanced root endpoint"""
    global bioclip_model, classifier
    
    model_status = "loaded" if all(m is not None for m in [bioclip_model, classifier]) else "not_loaded"
    
    return {
        "message": "BioCLIP Pest Identification API",
        "version": "4.0.0",
        "status": "running",
        "model_status": model_status,
        "startup_success": getattr(app.state, 'startup_success', False),
        "startup_error": getattr(app.state, 'startup_error', None),
        "uptime": time.time() - getattr(app.state, 'startup_time', time.time()),
        "endpoints": {
            "health": "/health - API health check",
            "predict": "/predict - Pest identification",
            "debug": "/debug - Debug information",
            "reload": "/reload - Reload models",
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
    """Comprehensive health check"""
    global bioclip_model, classifier, bioclip_preprocess
    
    models_status = {
        "bioclip_model": bioclip_model is not None,
        "classifier": classifier is not None,
        "preprocess": bioclip_preprocess is not None
    }
    
    health_data = {
        "timestamp": time.time(),
        "startup_success": getattr(app.state, 'startup_success', False),
        "startup_error": getattr(app.state, 'startup_error', None),
        "uptime": time.time() - getattr(app.state, 'startup_time', time.time()),
        "system_info": get_system_info(),
        "model_status": models_status,
        "file_status": {
            "classifier_exists": os.path.exists(LOCAL_MODEL_PATH),
            "classifier_size": os.path.getsize(LOCAL_MODEL_PATH) if os.path.exists(LOCAL_MODEL_PATH) else 0
        }
    }
    
    if not all(models_status.values()):
        return JSONResponse(
            status_code=503,
            content={
                "healthy": False,
                "status": "models_not_loaded",
                "message": "Models not loaded. Try /reload endpoint.",
                **health_data
            }
        )
    
    # Test model functionality
    try:
        dummy_input = torch.randn(1, 3, 224, 224).to(DEVICE)
        
        with torch.no_grad():
            start_time = time.time()
            features = bioclip_model.encode_image(dummy_input)
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
                "throughput_estimate": round(1.0 / inference_time, 1)
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
                **health_data
            }
        )

@app.post("/predict")
async def predict_pest(image: UploadFile = File(...)):
    """Enhanced pest prediction with validation"""
    global bioclip_model, classifier
    
    # Check model availability
    if any(m is None for m in [bioclip_model, classifier]):
        raise HTTPException(
            status_code=503,
            detail="Models not loaded. Check /health endpoint."
        )
    
    # Validate request
    if not image.content_type or not image.content_type.startswith('image/'):
        raise HTTPException(
            status_code=400,
            detail=f"Invalid file type: {image.content_type}"
        )
    
    try:
        # Read image
        image_bytes = await image.read()
        
        if len(image_bytes) == 0:
            raise HTTPException(status_code=400, detail="Empty image file")
        
        if len(image_bytes) > 20 * 1024 * 1024:  # 20MB limit
            raise HTTPException(status_code=400, detail="Image too large (max 20MB)")
        
        # Process image
        pil_image = Image.open(BytesIO(image_bytes))
        
        # Validate dimensions
        if min(pil_image.size) < 32:
            raise HTTPException(status_code=400, detail="Image too small (min 32x32)")
        
        # Convert to RGB
        if pil_image.mode != 'RGB':
            pil_image = pil_image.convert('RGB')
        
        # Preprocess
        image_tensor = preprocess_image(pil_image).to(DEVICE)
        
        # Predict
        start_time = time.time()
        
        with torch.no_grad():
            features = bioclip_model.encode_image(image_tensor)
            outputs = classifier(features)
            probabilities = F.softmax(outputs, dim=1)
            
            confidence, predicted_class = torch.max(probabilities, 1)
            predicted_idx = predicted_class.item()
            predicted_species = CLASS_NAMES[predicted_idx]
            
            # Get all scores
            all_probs = probabilities.cpu().numpy().flatten()
            class_scores = {
                class_name: round(float(prob), 4)
                for class_name, prob in zip(CLASS_NAMES, all_probs)
            }
            
            sorted_scores = dict(sorted(class_scores.items(), key=lambda x: x[1], reverse=True))
        
        processing_time = time.time() - start_time
        
        # Create result
        prediction_result = {
            "species": predicted_species,
            "confidence": round(confidence.item(), 4),
            "confidence_percentage": round(confidence.item() * 100, 2),
            "all_class_scores": sorted_scores,
            "prediction_quality": (
                "high" if confidence.item() > 0.8 else 
                "medium" if confidence.item() > 0.6 else 
                "low"
            )
        }
        
        # Create visualization
        try:
            annotated_image = create_prediction_visualization(pil_image, prediction_result)
        except Exception as e:
            logger.error(f"Visualization failed: {e}")
            annotated_image = None
        
        response = {
            "success": True,
            "prediction": prediction_result,
            "metadata": {
                "processing_time": round(processing_time, 4),
                "device": str(DEVICE),
                "image_size": pil_image.size,
                "file_size": len(image_bytes),
                "timestamp": time.time()
            }
        }
        
        if annotated_image:
            response["annotated_image"] = annotated_image
        
        return response
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/debug")
async def debug_info():
    """Enhanced debug information"""
    global bioclip_model, classifier, bioclip_preprocess
    
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
            "classifier_size": os.path.getsize(LOCAL_MODEL_PATH) if os.path.exists(LOCAL_MODEL_PATH) else 0
        },
        "configuration": {
            "class_names": CLASS_NAMES,
            "device": str(DEVICE),
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
            test_input = torch.randn(1, 3, 224, 224).to(DEVICE)
            with torch.no_grad():
                features = bioclip_model.encode_image(test_input)
                debug_data["model_details"] = {
                    "feature_dimension": features.shape[-1],
                    "input_shape": list(test_input.shape),
                    "feature_shape": list(features.shape)
                }
        except Exception as e:
            debug_data["model_details"] = {"error": str(e)}
    
    return debug_data

@app.post("/reload")
async def reload_models():
    """Enhanced model reload endpoint"""
    try:
        logger.info("🔄 Manual model reload requested")
        
        # Clear GPU cache
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        # Reload models
        start_time = time.time()
        await load_all_models()
        load_time = time.time() - start_time
        
        # Update app state
        app.state.startup_success = True
        app.state.startup_error = None
        app.state.startup_time = time.time()
        
        logger.info(f"✅ Models reloaded in {load_time:.2f}s")
        
        return {
            "success": True,
            "message": "Models reloaded successfully",
            "load_time": round(load_time, 2),
            "timestamp": time.time()
        }
        
    except Exception as e:
        logger.error(f"❌ Reload failed: {e}")
        
        app.state.startup_success = False
        app.state.startup_error = str(e)
        
        return JSONResponse(
            status_code=500,
            content={
                "success": False,
                "error": str(e),
                "timestamp": time.time()
            }
        )

# Error handlers
@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
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
    logger.error(f"Unhandled exception: {exc}")
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal server error",
            "message": str(exc),
            "timestamp": time.time()
        }
    )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=False)
