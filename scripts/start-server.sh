#!/bin/bash
# Startup script for the Mistral AI inference server optimized for H100 GPUs

set -e

# Print environment for debugging
echo "============== Server Configuration =============="
echo "Model ID: ${MODEL_ID}"
echo "Tensor Parallel Size: ${TENSOR_PARALLEL_SIZE}"
echo "GPU Memory Utilization: ${GPU_MEMORY_UTILIZATION}"
echo "Max Model Length: ${MAX_MODEL_LEN}"
echo "Data Type: ${DTYPE}"
echo "Monitoring Enabled: ${ENABLE_MONITORING}"
echo "=============================================="

# Check for NVIDIA GPUs
nvidia-smi --query-gpu=name,memory.total --format=csv

# Wait for any dependencies (if needed)
if [ -n "$OTLP_ENDPOINT" ]; then
  echo "Waiting for SigNoz OTLP endpoint at $OTLP_ENDPOINT to be available..."
  # Simple connection check - can be enhanced for production
  timeout=60
  while ! nc -z $(echo $OTLP_ENDPOINT | cut -d/ -f3 | cut -d: -f1) $(echo $OTLP_ENDPOINT | cut -d: -f3 | cut -d/ -f1) && [ $timeout -gt 0 ]; do
    timeout=$((timeout-1))
    sleep 1
    echo "Waiting for SigNoz OTLP endpoint... ($timeout seconds remaining)"
  done
fi

# Start the FastAPI application
echo "Starting Mistral AI inference server..."
exec python3 -m app.main
