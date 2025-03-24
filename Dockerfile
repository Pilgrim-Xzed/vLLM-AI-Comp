FROM nvidia/cuda:12.2.0-devel-ubuntu22.04 as base

# Set up environment
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 \
    python3-pip \
    python3-dev \
    git \
    wget \
    curl \
    build-essential \
    ninja-build \
    && rm -rf /var/lib/apt/lists/*

# Create a non-root user
RUN useradd -m -u 1000 vllm
USER vllm
WORKDIR /home/vllm

# Set up Python environment
ENV PATH="/home/vllm/.local/bin:$PATH"
RUN python3 -m pip install --no-cache-dir --upgrade pip

# Install PyTorch with CUDA 12.1 support (optimized for H100)
RUN pip install --no-cache-dir torch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2 --index-url https://download.pytorch.org/whl/cu121

# Copy requirements and install dependencies
COPY --chown=vllm:vllm requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Create directories for the application
RUN mkdir -p /home/vllm/app

# Copy the application code
COPY --chown=vllm:vllm app /home/vllm/app
COPY --chown=vllm:vllm config /home/vllm/config
COPY --chown=vllm:vllm scripts /home/vllm/scripts

# Make scripts executable
RUN chmod +x /home/vllm/scripts/*.sh

# Expose the server port
EXPOSE 8000

# Set entrypoint to the startup script
ENTRYPOINT ["/home/vllm/scripts/start-server.sh"]
