# Mistral AI Inference Server

A production-ready Mistral AI inference server optimized for 2xH100 GPUs using vLLM. This implementation provides an OpenAI-compatible API for high-throughput, low-latency inferencing.

## Architecture

This project implements a high-performance inference server with the following features:

- **Tensor Parallelism**: Optimized for 2xH100 GPUs to maximize throughput
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

- **Model Support**: Mistral AI models (default: Mistral-7B-Instruct-v0.1)
- **Framework**: vLLM with PyTorch
- **GPU Requirements**: 2x NVIDIA H100 GPUs
- **Optimization**: BF16 precision, flash attention, tensor parallelism
- **Monitoring**: SigNoz (OpenTelemetry)
- **API**: FastAPI with async endpoints

## Getting Started

### Prerequisites

- Docker and Docker Compose
- 2x NVIDIA H100 GPUs
- NVIDIA Docker Runtime

### Configuration

1. Copy the sample environment file:
   ```
   cp config/sample.env .env
   ```

2. Edit the `.env` file to configure your deployment:
   - Set `MODEL_ID` to your desired Mistral model
   - Configure `HUGGING_FACE_HUB_TOKEN` if using gated models
   - Adjust `TENSOR_PARALLEL_SIZE`, `GPU_MEMORY_UTILIZATION` and other parameters as needed

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

This implementation uses SigNoz for monitoring instead of Prometheus/Grafana. To enable monitoring:

1. Deploy SigNoz following their [official documentation](https://signoz.io/docs/install/)
2. Configure the OpenTelemetry endpoint in `.env` to point to your SigNoz instance
3. View traces, metrics, and logs in the SigNoz dashboard

## Performance Optimization

This server is optimized for performance on H100 GPUs:

- Uses tensor parallelism across 2 GPUs
- Employs BFloat16 precision for optimal performance
- Utilizes Flash Attention for faster transformer computations
- Implements efficient batch processing for high throughput

## Client Example

```python
import requests

url = "http://localhost:8000/v1/chat/completions"
headers = {"Content-Type": "application/json"}
data = {
    "model": "mistralai/Mistral-7B-Instruct-v0.1",
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
