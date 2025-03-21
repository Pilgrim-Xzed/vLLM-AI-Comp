#!/bin/bash
set -e

echo "Starting Mistral Inference Server with vLLM on 2xH100 GPUs"

# Check if .env file exists
if [ ! -f .env ]; then
    echo "Creating .env file from .env.example..."
    cp .env.example .env
    echo "Please edit .env file with your Hugging Face token before proceeding."
    echo "You can do this by running: nano .env"
    exit 1
fi

# Check if HF token is set
grep -q "your-hf-token-here" .env
if [ $? -eq 0 ]; then
    echo "Error: You need to set HUGGING_FACE_HUB_TOKEN in the .env file"
    echo "Please edit the .env file and replace 'your-hf-token-here' with your actual token"
    exit 1
fi

# Start the server using Docker Compose
echo "Starting server with Docker Compose..."
docker compose up -d

# Wait for server to be ready
echo "Waiting for server to be ready..."
timeout=120
elapsed=0
while [ $elapsed -lt $timeout ]; do
    if curl -s http://localhost:8000/health > /dev/null; then
        echo "Server is ready!"
        echo "API is available at: http://localhost:8000"
        echo "Prometheus metrics at: http://localhost:8000/metrics"
        echo "Grafana dashboard at: http://localhost:3000"
        echo ""
        echo "To test the server, run: python client_example.py --user \"Tell me about quantum computing\""
        exit 0
    fi
    sleep 5
    elapsed=$((elapsed+5))
    echo "Still waiting... ($elapsed seconds elapsed)"
done

echo "Server did not start within the timeout period. Check logs with 'docker compose logs'"
echo "You can manually check if it's running with 'curl http://localhost:8000/health'"
exit 1
