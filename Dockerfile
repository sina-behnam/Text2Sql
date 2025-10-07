# Use NVIDIA CUDA base image for GPU support
FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    LANG=C.UTF-8 \
    LC_ALL=C.UTF-8

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3.12.3 \
    python3-pip \
    git \
    wget \
    curl \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip
RUN pip3 install --upgrade pip setuptools wheel

# Copy requirements file
COPY requirements.txt /app/requirements.txt

# Install Python dependencies
# Install PyTorch with CUDA 11.8 support first
RUN pip3 install torch==2.5.0+cu118 torchvision==0.20.0+cu118 torchaudio==2.5.0+cu118 \
    --extra-index-url https://download.pytorch.org/whl/cu118

# Install other dependencies
RUN pip3 install -r requirements.txt

# Download SpaCy model
RUN python3 -m spacy download en_core_web_trf

# Copy project files
COPY src/ /app/src/
COPY pipeline/ /app/pipeline/
COPY tests/ /app/tests/
COPY main.py /app/main.py
COPY .gitignore /app/.gitignore

# Create necessary directories
RUN mkdir -p /app/Data \
    /app/models \
    /app/databases \
    /app/output \
    /app/logs

# Set permissions
RUN chmod +x /app/main.py

# Default command
CMD ["python3", "main.py", "--help"]