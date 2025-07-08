# ===============================
# --- Builder Stage ---
# ===============================
FROM nvidia/cuda:11.8.0-base-ubuntu20.04 AS builder

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3.9 \
    python3.9-dev \
    python3-pip \
    python3.9-venv \
    python3.9-distutils \
    git \
    wget \
    curl \
    build-essential \
    cmake \
    pkg-config \
    libopencv-dev \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    libglu1-mesa \
    libxi6 \
    libxmu6 \
    libgstreamer1.0-0 \
    libfontconfig1 \
    libfreetype6 \
    libssl-dev \
    libffi-dev \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Create and activate virtual environment
RUN python3.9 -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Upgrade pip and install build tools
RUN pip install --no-cache-dir --upgrade pip setuptools wheel

# Copy requirements and install dependencies
COPY requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt

# ===============================
# --- Production Stage ---
# ===============================
FROM nvidia/cuda:11.8.0-base-ubuntu20.04

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PATH="/opt/venv/bin:$PATH" \
    TORCH_HOME=/app/.torch \
    HF_HOME=/app/.huggingface \
    TRANSFORMERS_CACHE=/app/.huggingface \
    PORT=8000

# Install runtime dependencies
RUN apt-get update && apt-get install -y \
    python3.9 \
    python3.9-venv \
    python3.9-distutils \
    libopencv-dev \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    libglu1-mesa \
    libxi6 \
    libxmu6 \
    libgstreamer1.0-0 \
    libfontconfig1 \
    libfreetype6 \
    fonts-liberation \
    fonts-dejavu-core \
    fonts-dejavu-extra \
    curl \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Copy virtual environment from builder
COPY --from=builder /opt/venv /opt/venv

# Create app directory
WORKDIR /app

# Add non-root user
RUN groupadd -r appuser && useradd -r -g appuser -u 1001 appuser

# Create cache and log dirs and set permissions
RUN mkdir -p /app/.torch /app/.huggingface /app/models /app/logs \
    && chown -R appuser:appuser /app

# Copy app files
COPY --chown=appuser:appuser main.py /app/
COPY --chown=appuser:appuser requirements.txt /app/
COPY --chown=appuser:appuser best_bioclip_classifier.pth /app/

# Check if critical files exist
RUN if [ ! -f "/app/main.py" ]; then echo "ERROR: main.py not found" && exit 1; fi
RUN if [ ! -f "/app/best_bioclip_classifier.pth" ]; then \
        echo "WARNING: Model file not found. Service may not work correctly."; \
    fi

# Fix permissions
RUN chown -R appuser:appuser /app

# Use non-root user
USER appuser

# Pre-download BioCLIP model (optional)
RUN python3.9 -c "import open_clip; print('Testing BioCLIP model loading...'); \
    try: \
        model, _, _ = open_clip.create_model_and_transforms('hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224'); \
        print('BioCLIP model pre-downloaded successfully.'); \
    except Exception as e: \
        print(f'Error pre-downloading BioCLIP model: {e}'); exit(1)"

# Expose port
EXPOSE ${PORT}

# Run FastAPI app using Uvicorn
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
