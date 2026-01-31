# EVOCOMP Docker Image
# Base image with CUDA support
FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3.10 \
    python3.10-dev \
    python3-pip \
    python3.10-venv \
    git \
    wget \
    curl \
    build-essential \
    cmake \
    libopenmpi-dev \
    openmpi-bin \
    redis-server \
    && rm -rf /var/lib/apt/lists/*

# Set Python 3.10 as default
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.10 1

# Create non-root user
RUN useradd -m -u 1000 -s /bin/bash evocomp
USER evocomp
WORKDIR /home/evocomp/app

# Copy requirements first for better caching
COPY --chown=evocomp:evocomp requirements.txt .
COPY --chown=evocomp:evocomp requirements-dev.txt .

# Create virtual environment
RUN python -m venv /home/evocomp/venv
ENV PATH="/home/evocomp/venv/bin:$PATH"

# Install Python dependencies
RUN pip install --upgrade pip && \
    pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118 && \
    pip install -r requirements.txt

# Copy source code
COPY --chown=evocomp:evocomp src/ ./src/
COPY --chown=evocomp:evocomp scripts/ ./scripts/
COPY --chown=evocomp:evocomp configs/ ./configs/
COPY --chown=evocomp:evocomp examples/ ./examples/

# Install package in development mode
RUN pip install -e .

# Create necessary directories
RUN mkdir -p /home/evocomp/app/logs /home/evocomp/app/data /home/evocomp/app/models

# Expose ports
EXPOSE 6379  # Redis
EXPOSE 9090  # Prometheus
EXPOSE 3000  # Grafana
