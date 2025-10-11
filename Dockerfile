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
    python3.10 \
    python3-pip \
    git \
    wget \
    curl \
    build-essential \
    openssh-server \
    && rm -rf /var/lib/apt/lists/*

# Configure SSH AFTER installing openssh-server
RUN mkdir -p /var/run/sshd && \
    sed -i 's/#PermitRootLogin prohibit-password/PermitRootLogin yes/' /etc/ssh/sshd_config

# Upgrade pip
RUN pip3 install --upgrade pip setuptools wheel

# Copy requirements file
COPY requirements-docker.txt /app/requirements.txt

# Install PyTorch with CUDA 11.8 support first
RUN pip3 install torch==2.5.0+cu118 torchvision==0.20.0+cu118 torchaudio==2.5.0+cu118 \
    --extra-index-url https://download.pytorch.org/whl/cu118

# Install other dependencies + Jupyter + ipykernel
RUN pip3 install -r requirements.txt && \
    pip3 install kaggle && \
    pip3 install jupyter jupyterlab notebook ipykernel && \
    python3 -m ipykernel install --user --name=python3 --display-name="Python 3"

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

# Expose ports
EXPOSE 8888 22

# Create Jupyter config
RUN mkdir -p /root/.jupyter && \
    echo "c.ServerApp.allow_origin = '*'\n\
c.ServerApp.disable_check_xsrf = True\n\
c.NotebookApp.allow_origin = '*'\n\
c.NotebookApp.disable_check_xsrf = True\n\
c.ServerApp.ip = '0.0.0.0'\n\
c.ServerApp.port = 8888\n\
c.ServerApp.open_browser = False\n\
c.ServerApp.allow_root = True\n\
c.ServerApp.token = ''\n\
c.ServerApp.password = ''\n\
c.LabApp.tornado_settings = {'headers': {'Content-Security-Policy': \"frame-ancestors * 'self'\"}}\n\
" > /root/.jupyter/jupyter_lab_config.py

# Create startup script
RUN echo '#!/bin/bash\n\
set -e\n\
service ssh start\n\
echo "SSH started on port 22"\n\
echo "Starting Jupyter Lab on port 8888..."\n\
exec jupyter lab\n\
' > /start.sh && chmod +x /start.sh

# Use the startup script as CMD
CMD ["/bin/bash", "/start.sh"]