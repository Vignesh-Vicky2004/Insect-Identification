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
import json
import os
import cv2
from typing import Dict, Any, List
import logging
import time

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create FastAPI app instance
app = FastAPI(
    title="BioCLIP Pest Identification API",
    version="2.0.0",
    description="AI-powered pest identification using BioCLIP foundation model"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Model configuration
class ModelConfig:
    MODEL_PATH = "best_bioclip_classifier.pth"
    BIOCLIP_MODEL = 'hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224'
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Class names from your training
    CLASS_NAMES = [
        'Leaf-folder',
        'Pink-Bollworm',
        'leaf folder - adult',
        'stem borer - adult',
        'stemborer'
    ]

    # Confidence threshold
    CONFIDENCE_THRESHOLD = 0.5


# BioCLIP Classifier Architecture (same as training)
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


class BioCLIPPestIdentificationService:
    def __init__(self):
        self.device = ModelConfig.DEVICE
        self.class_names = ModelConfig.CLASS_NAMES
        self.confidence_threshold = ModelConfig.CONFIDENCE_THRESHOLD

        # Load models
        self._load_bioclip_model()
        self._load_classifier()
        self._setup_transforms()

        logger.info(f"BioCLIP Pest Identification Service initialized on {self.device}")

    def _load_bioclip_model(self):
        """Load the BioCLIP foundation model"""
        try:
            logger.info("Loading BioCLIP foundation model...")
            self.bioclip_model, _, _ = open_clip.create_model_and_transforms(
                ModelConfig.BIOCLIP_MODEL
            )
            self.bioclip_model = self.bioclip_model.to(self.device)
            self.bioclip_model.eval()

            # Get output dimension
            dummy_input = torch.randn(1, 3, 224, 224).to(self.device)
            with torch.no_grad():
                features = self.bioclip_model.encode_image(dummy_input)
                self.feature_dim = features.shape[-1]

            logger.info(f"BioCLIP model loaded successfully. Feature dimension: {self.feature_dim}")

        except Exception as e:
            logger.error(f"Failed to load BioCLIP model: {e}")
            raise HTTPException(status_code=500, detail=f"Model loading failed: {e}")

    def _load_classifier(self):
        """Load the trained classifier"""
        try:
            if not os.path.exists(ModelConfig.MODEL_PATH):
                raise FileNotFoundError(f"Model file not found: {ModelConfig.MODEL_PATH}")

            logger.info(f"Loading trained classifier from {ModelConfig.MODEL_PATH}...")
            self.classifier = OptimizedClassifier(
                input_dim=self.feature_dim,
                num_classes=len(self.class_names)
            ).to(self.device)

            # Load trained weights
            state_dict = torch.load(ModelConfig.MODEL_PATH, map_location=self.device)
            self.classifier.load_state_dict(state_dict)
            self.classifier.eval()

            logger.info("Classifier loaded successfully")

        except Exception as e:
            logger.error(f"Failed to load classifier: {e}")
            raise HTTPException(status_code=500, detail=f"Classifier loading failed: {e}")

    def _setup_transforms(self):
        """Setup image preprocessing transforms"""
        self.transform = A.Compose([
            A.Resize(224, 224),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2(),
        ])

    def _preprocess_image(self, image: Image.Image) -> torch.Tensor:
        """Preprocess image for model inference"""
        try:
            # Convert PIL to numpy array
            image_np = np.array(image.convert('RGB'))

            # Apply transforms
            transformed = self.transform(image=image_np)
            image_tensor = transformed['image']

            # Add batch dimension
            image_tensor = image_tensor.unsqueeze(0).to(self.device)

            return image_tensor

        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Image preprocessing failed: {e}")

    def _predict(self, image_tensor: torch.Tensor) -> Dict[str, Any]:
        """Run model inference"""
        try:
            start_time = time.time()

            with torch.no_grad():
                # Extract features using BioCLIP
                features = self.bioclip_model.encode_image(image_tensor)

                # Classify using trained classifier
                outputs = self.classifier(features)
                probabilities = F.softmax(outputs, dim=1)

                # Get predictions
                confidence, predicted_class = torch.max(probabilities, 1)

                confidence_score = confidence.item()
                predicted_idx = predicted_class.item()
                predicted_class_name = self.class_names[predicted_idx]

                # Get all class probabilities
                all_probabilities = probabilities.squeeze().cpu().numpy()

            processing_time = time.time() - start_time

            return {
                'predicted_class': predicted_class_name,
                'confidence': confidence_score,
                'class_probabilities': {
                    class_name: float(prob)
                    for class_name, prob in zip(self.class_names, all_probabilities)
                },
                'processing_time': processing_time
            }

        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Model inference failed: {e}")

    def create_annotated_image(self, image: Image.Image, prediction: Dict[str, Any]) -> str:
        """Create annotated image with prediction results"""
        try:
            # Create a copy of the image
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
                        font = ImageFont.truetype(font_path, 24)
                        break
                    except:
                        continue
                if font is None:
                    font = ImageFont.load_default()
            except:
                font = ImageFont.load_default()

            # Create prediction text
            predicted_class = prediction['predicted_class']
            confidence = prediction['confidence']
            confidence_percentage = confidence * 100

            # Main prediction label
            main_label = f"{predicted_class}"
            confidence_label = f"Confidence: {confidence_percentage:.1f}%"

            # Get image dimensions
            img_width, img_height = annotated_image.size

            # Calculate text dimensions
            main_bbox = draw.textbbox((0, 0), main_label, font=font)
            conf_bbox = draw.textbbox((0, 0), confidence_label, font=font)

            main_text_width = main_bbox[2] - main_bbox[0]
            main_text_height = main_bbox[3] - main_bbox[1]
            conf_text_width = conf_bbox[2] - conf_bbox[0]
            conf_text_height = conf_bbox[3] - conf_bbox[1]

            # Position for text (top-left corner with padding)
            padding = 10
            x_pos = padding
            y_pos = padding

            # Background rectangle for main label
            bg_width = max(main_text_width, conf_text_width) + 20
            bg_height = main_text_height + conf_text_height + 30

            # Choose color based on confidence
            if confidence >= 0.8:
                bg_color = "green"
                text_color = "white"
            elif confidence >= 0.6:
                bg_color = "orange"
                text_color = "white"
            else:
                bg_color = "red"
                text_color = "white"

            # Draw background rectangle
            draw.rectangle([x_pos, y_pos, x_pos + bg_width, y_pos + bg_height],
                           fill=bg_color, outline=bg_color)

            # Draw main label text
            draw.text((x_pos + 10, y_pos + 5), main_label, fill=text_color, font=font)

            # Draw confidence text
            draw.text((x_pos + 10, y_pos + main_text_height + 15), confidence_label,
                      fill=text_color, font=font)

            # Add a border around the entire image
            border_width = 3
            draw.rectangle([0, 0, img_width - 1, img_height - 1],
                           outline=bg_color, width=border_width)

            # Convert to base64
            buffer = BytesIO()
            annotated_image.save(buffer, format='JPEG', quality=95)
            img_str = base64.b64encode(buffer.getvalue()).decode()

            return img_str

        except Exception as e:
            logger.error(f"Error creating annotation: {e}")
            # Return original image if annotation fails
            buffer = BytesIO()
            image.save(buffer, format='JPEG', quality=95)
            img_str = base64.b64encode(buffer.getvalue()).decode()
            return img_str

    async def identify_pest(self, image_file: UploadFile) -> Dict[str, Any]:
        """Main identification function"""
        try:
            # Read and validate image
            image_bytes = await image_file.read()

            try:
                image = Image.open(BytesIO(image_bytes))
                if image.mode != 'RGB':
                    image = image.convert('RGB')
            except Exception as e:
                raise HTTPException(status_code=400, detail=f"Invalid image format: {e}")

            # Preprocess image
            image_tensor = self._preprocess_image(image)

            # Run prediction
            prediction_result = self._predict(image_tensor)

            # Create annotated image
            annotated_base64 = self.create_annotated_image(image, prediction_result)

            # Check if prediction meets confidence threshold
            is_confident = prediction_result['confidence'] >= self.confidence_threshold

            # Format response
            formatted_response = {
                "success": True,
                "model_info": {
                    "model_type": "BioCLIP",
                    "version": "2.0.0",
                    "confidence_threshold": self.confidence_threshold
                },
                "prediction": {
                    "species": prediction_result['predicted_class'],
                    "confidence": round(prediction_result['confidence'], 4),
                    "confidence_percentage": round(prediction_result['confidence'] * 100, 2),
                    "is_confident": is_confident,
                    "status": "confident" if is_confident else "uncertain"
                },
                "all_predictions": [
                    {
                        "species": class_name,
                        "confidence": round(prob, 4),
                        "confidence_percentage": round(prob * 100, 2)
                    }
                    for class_name, prob in prediction_result['class_probabilities'].items()
                ],
                "image_info": {
                    "width": image.size[0],
                    "height": image.size[1],
                    "format": image.format or "JPEG"
                },
                "annotated_image": annotated_base64,
                "processing_time": round(prediction_result['processing_time'], 3),
                "device_used": str(self.device)
            }

            return formatted_response

        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Identification failed: {e}")
            raise HTTPException(status_code=500, detail=f"Identification failed: {e}")


# Initialize service
try:
    pest_service = BioCLIPPestIdentificationService()
    SERVICE_READY = True
    logger.info("Service initialized successfully")
except Exception as e:
    SERVICE_READY = False
    logger.error(f"Service initialization failed: {e}")


# Routes
@app.get("/")
async def root():
    return {
        "message": "BioCLIP Pest Identification API",
        "status": "running" if SERVICE_READY else "error",
        "version": "2.0.0",
        "model": "BioCLIP + Optimized Classifier",
        "device": str(ModelConfig.DEVICE),
        "classes": ModelConfig.CLASS_NAMES,
        "endpoints": {
            "identify": "/identify (POST)",
            "health": "/health",
            "model_info": "/model-info",
            "docs": "/docs"
        }
    }


@app.get("/health")
async def health_check():
    if not SERVICE_READY:
        raise HTTPException(status_code=503, detail="Service not ready")

    return {
        "status": "healthy",
        "service": "bioclip-pest-identification",
        "model_loaded": True,
        "device": str(ModelConfig.DEVICE),
        "classes_count": len(ModelConfig.CLASS_NAMES)
    }


@app.get("/model-info")
async def model_info():
    if not SERVICE_READY:
        raise HTTPException(status_code=503, detail="Service not ready")

    return {
        "model_type": "BioCLIP + Custom Classifier",
        "foundation_model": ModelConfig.BIOCLIP_MODEL,
        "classifier_path": ModelConfig.MODEL_PATH,
        "device": str(ModelConfig.DEVICE),
        "classes": ModelConfig.CLASS_NAMES,
        "confidence_threshold": ModelConfig.CONFIDENCE_THRESHOLD,
        "input_size": "224x224",
        "feature_dimension": pest_service.feature_dim if SERVICE_READY else None
    }


@app.post("/identify")
async def identify_pest(image: UploadFile = File(...)):
    """Upload an image and get pest identification results using BioCLIP"""

    if not SERVICE_READY:
        raise HTTPException(status_code=503, detail="Service not ready - model loading failed")

    # Validate file
    if not image.content_type or not image.content_type.startswith('image/'):
        raise HTTPException(status_code=400, detail="File must be an image")

    if image.size and image.size > 10 * 1024 * 1024:  # 10MB limit
        raise HTTPException(status_code=400, detail="Image file too large (max 10MB)")

    try:
        result = await pest_service.identify_pest(image)
        return JSONResponse(content=result)
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


# Additional endpoint for batch processing
@app.post("/identify-batch")
async def identify_pest_batch(images: List[UploadFile] = File(...)):
    """Upload multiple images for batch pest identification"""

    if not SERVICE_READY:
        raise HTTPException(status_code=503, detail="Service not ready")

    if len(images) > 10:
        raise HTTPException(status_code=400, detail="Maximum 10 images per batch")

    results = []

    for i, image in enumerate(images):
        try:
            if not image.content_type or not image.content_type.startswith('image/'):
                results.append({
                    "image_index": i,
                    "filename": image.filename,
                    "success": False,
                    "error": "Invalid image format"
                })
                continue

            result = await pest_service.identify_pest(image)
            result["image_index"] = i
            result["filename"] = image.filename
            results.append(result)

        except Exception as e:
            results.append({
                "image_index": i,
                "filename": image.filename,
                "success": False,
                "error": str(e)
            })

    return JSONResponse(content={
        "success": True,
        "batch_size": len(images),
        "results": results
    })


# Error handler
@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    logger.error(f"Unhandled exception: {exc}")
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error", "type": "UnhandledException"}
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
