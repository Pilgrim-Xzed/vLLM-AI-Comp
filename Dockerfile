FROM nvidia/cuda:12.1.1-devel-ubuntu22.04 as base

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
    && rm -rf /var/lib/apt/lists/*

# Create a non-root user
RUN useradd -m -u 1000 vllm
USER vllm
WORKDIR /home/vllm

# Set up Python environment
ENV PATH="/home/vllm/.local/bin:$PATH"
RUN python3 -m pip install --no-cache-dir --upgrade pip

# Copy requirements and install dependencies
COPY --chown=vllm:vllm requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the code
COPY --chown=vllm:vllm . .

# Expose the server port
EXPOSE 8000

# Run the server
CMD ["python3", "server.py"]
