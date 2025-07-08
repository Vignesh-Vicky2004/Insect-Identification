# Multi-stage build for optimized BioCLIP pest identification service

# --- Builder Stage ---
FROM nvidia/cuda:11.8-base-ubuntu20.04 as builder

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# Install system dependencies for building
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

# Copy requirements first for better caching
COPY requirements.txt /tmp/requirements.txt

# Install Python dependencies
RUN pip install --no-cache-dir -r /tmp/requirements.txt

# --- Production Stage ---
FROM nvidia/cuda:11.8-base-ubuntu20.04

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV PATH="/opt/venv/bin:$PATH"
ENV TORCH_HOME=/app/.torch
ENV HF_HOME=/app/.huggingface
ENV TRANSFORMERS_CACHE=/app/.huggingface
ENV PORT=8000

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

# Copy virtual environment from builder stage
COPY --from=builder /opt/venv /opt/venv

# Create app directory
WORKDIR /app

# Create non-root user for security
RUN groupadd -r appuser && useradd -r -g appuser -u 1001 appuser

# Create directories for model caching and logs, and set permissions
RUN mkdir -p /app/.torch /app/.huggingface /app/models /app/logs \
    && chown -R appuser:appuser /app

# Copy application files
COPY --chown=appuser:appuser main.py /app/
COPY --chown=appuser:appuser requirements.txt /app/

# Copy model file if it exists (critical for the service)
COPY --chown=appuser:appuser best_bioclip_classifier.pth /app/

# Validate critical files exist
RUN if [ ! -f "/app/main.py" ]; then echo "ERROR: main.py not found" && exit 1; fi
RUN if [ ! -f "/app/best_bioclip_classifier.pth" ]; then \
        echo "WARNING: Model file not found. Service may not work correctly."; \
    fi

# Set proper permissions for the app directory
RUN chown -R appuser:appuser /app

# Switch to non-root user
USER appuser

# Pre-download BioCLIP model (optional - reduces startup time)
# This will download the model to the TORCH_HOME/HF_HOME paths during build
RUN python3.9 -c "import open_clip; print('Testing BioCLIP model loading...'); \
    try: \
        model, _, _ = open_clip.create_model_and_transforms('hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224'); \
        print('BioCLIP model pre-downloaded successfully.'); \
    except Exception as e: \
        print(f'Error pre-downloading BioCLIP model: {e}'); \
        exit(1)"

# Expose the port FastAPI runs on
EXPOSE ${PORT}

# Command to run the application using Uvicorn
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
