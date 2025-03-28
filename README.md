# Llama 8B Quantized Inference Server

A high-performance Llama 8B inference server optimized for throughput (2500-3000 tokens/sec) using AWQ quantization with vLLM. This implementation provides an OpenAI-compatible API for efficient, low-latency inferencing.

## Architecture

This project implements a high-throughput inference server with the following features:

- **AWQ Quantization**: 4-bit quantization for optimal memory usage and performance
- **High Throughput**: Optimized for 2500-3000 tokens/sec output
- **OpenAI-compatible API**: Drop-in replacement for OpenAI API clients
- **SigNoz Monitoring**: Comprehensive observability with distributed tracing
- **Production-Ready**: Designed for reliability, scalability, and performance

### Directory Structure

```
.
├── app/                      # Application code
│   ├── api/                  # API endpoints
│   ├── core/                 # Core model handling 
│   ├── models/               # Model-specific code
│   ├── monitoring/           # SigNoz telemetry integration
│   ├── schemas/              # API schemas
│   └── utils/                # Utility functions
├── config/                   # Configuration files
│   ├── otel-collector-config.yaml  # OpenTelemetry collector config
│   └── sample.env            # Sample environment variables
├── scripts/                  # Operational scripts
│   └── start-server.sh       # Server startup script
├── tests/                    # Test suite
├── .env                      # Environment variables (gitignored)
├── Dockerfile                # Container definition
├── docker-compose.yml        # Service orchestration
└── requirements.txt          # Python dependencies
```

## Technical Specifications

- **Model**: Meta-Llama-3-8B (quantized with AWQ)
- **Framework**: vLLM 0.4.0 with PyTorch
- **GPU Requirements**: Single NVIDIA GPU with at least 24GB VRAM
- **Optimization**: 
  - AWQ 4-bit quantization
  - Optimized KV cache management with block size 16
  - 95% GPU memory utilization
  - Increased batch processing for higher throughput
- **Monitoring**: SigNoz (OpenTelemetry)
- **API**: FastAPI with async endpoints
- **Performance**: 2500-3000 tokens/second throughput

## Getting Started

### Prerequisites

- Docker and Docker Compose
- NVIDIA GPU with at least 24GB VRAM
- NVIDIA Docker Runtime

### Configuration

1. Copy the sample environment file:
   ```
   cp config/sample.env .env
   ```

2. Edit the `.env` file to configure your deployment:
   - Verify the `MODEL_ID` is set to Meta-Llama-3-8B
   - Configure `HUGGING_FACE_HUB_TOKEN` for accessing the Llama models
   - Adjust `QUANTIZATION`, `GPU_MEMORY_UTILIZATION` and other parameters as needed

### Deployment

Start the server:

```bash
docker-compose up -d
```

The server will be available at http://localhost:8000 with the following endpoints:

- `/v1/chat/completions` - Chat completion API
- `/v1/completions` - Text completion API
- `/v1/models` - List available models
- `/health` - Health check endpoint

## Monitoring with SigNoz

This implementation uses SigNoz for monitoring. To enable monitoring:

1. Deploy SigNoz following their [official documentation](https://signoz.io/docs/install/)
2. Configure the OpenTelemetry endpoint in `.env` to point to your SigNoz instance
3. View traces, metrics, and logs in the SigNoz dashboard

## Performance Optimization Details

The server is highly optimized for throughput with Llama 8B:

- **AWQ Quantization**: Reduces memory footprint while maintaining model quality
- **Single GPU Optimization**: For quantized models, using a single GPU is more efficient than tensor parallelism
- **Batch Processing**: Configured for maximum batch throughput
- **KV Cache Management**: Optimized block size and swap space for efficient memory usage
- **Server Configuration**: Optimized uvicorn settings for better request handling
- **Memory Utilization**: Increased to 95% for maximum performance

## Client Example

```python
import requests

url = "http://localhost:8000/v1/chat/completions"
headers = {"Content-Type": "application/json"}
data = {
    "model": "meta-llama/Meta-Llama-3-8B",
    "messages": [
        {"role": "system", "content": "You are a helpful AI assistant."},
        {"role": "user", "content": "Write a short poem about machine learning."}
    ],
    "temperature": 0.7,
    "max_tokens": 500
}

response = requests.post(url, json=data, headers=headers)
print(response.json())
```

## License

[MIT License](LICENSE)
