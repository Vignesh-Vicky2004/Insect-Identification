FROM python:3.10-slim

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV DEBIAN_FRONTEND=noninteractive

# Install system dependencies
RUN apt-get update && apt-get install -y \
    curl \
    wget \
    git \
    build-essential \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Create working directory
WORKDIR /app

# Create requirements.txt with all necessary dependencies
RUN echo "fastapi==0.104.1" > requirements.txt && \
    echo "uvicorn[standard]==0.24.0" >> requirements.txt && \
    echo "python-multipart==0.0.6" >> requirements.txt && \
    echo "torch==2.1.0" >> requirements.txt && \
    echo "torchvision==0.16.0" >> requirements.txt && \
    echo "open-clip-torch==2.20.0" >> requirements.txt && \
    echo "transformers==4.35.0" >> requirements.txt && \
    echo "Pillow==10.0.1" >> requirements.txt && \
    echo "requests==2.31.0" >> requirements.txt && \
    echo "numpy==1.24.3" >> requirements.txt

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code (rename your paste.txt to main.py)
COPY main.py .

# Create model cache directory
RUN mkdir -p ./model_cache

# Create non-root user
RUN useradd -m -u 1000 appuser && chown -R appuser:appuser /app
USER appuser

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=120s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Expose port
EXPOSE 8000

# Run the application
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
