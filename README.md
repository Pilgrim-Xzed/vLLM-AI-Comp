# Mistral Inference Server with vLLM

A production-grade OpenAI-compatible inference server for hosting Mistral models on 2xH100 GPUs using vLLM.

## Features

- OpenAI-compatible API endpoints (chat completions, completions)
- Optimized for high throughput on 2xH100 GPUs
- Tensorized inference with vLLM for maximum GPU utilization
- Request batching and continuous batching for efficiency
- Prometheus metrics for monitoring
- OpenTelemetry for distributed tracing
- Paged Attention for memory-efficient inference
- Health checks and graceful shutdowns

## Quick Start

1. Install dependencies (for local Python development):
   ```bash
   pip install -r requirements.txt
   ```

2. Configure your environment:
   ```bash
   cp .env.example .env
   # Edit .env file with your settings including your Hugging Face token
   ```

3. Start the server (Python method):
   ```bash
   python server.py
   ```

4. Recommended: For production deployment using official vLLM Docker image:
   ```bash
   docker compose up -d
   ```
   
   This uses the official `vllm/vllm-openai:latest` image from Mistral AI with proper configuration.

## API Documentation

Access the API documentation at `http://localhost:8000/docs` after starting the server.

## Configuration

The server can be configured via environment variables or the `.env` file:

- `MODEL_ID`: Hugging Face model ID for Mistral (default: "mistralai/mistral-7b")
- `GPU_MEMORY_UTILIZATION`: Target GPU memory utilization (default: 0.9)
- `MAX_MODEL_LEN`: Maximum sequence length (default: 8192)
- `PORT`: Server port (default: 8000)
- `HOST`: Server host (default: "0.0.0.0")
- `TENSOR_PARALLEL_SIZE`: Number of GPUs for tensor parallelism (default: 2 for 2xH100)
- `HUGGING_FACE_HUB_TOKEN`: Your Hugging Face token for accessing models (required for Docker deployment)

## Docker Deployment

The `docker-compose.yml` uses the official `vllm/vllm-openai:latest` image as recommended by Mistral AI. It's configured for:
- Tensor parallelism across 2 GPUs
- Model loading from Hugging Face (requires authentication token)
- Proper configuration for Mistral models with optimized parameters

To run:
```bash
docker compose up -d
```

## Monitoring

Prometheus metrics are exposed at `/metrics` endpoint.

## License

MIT
