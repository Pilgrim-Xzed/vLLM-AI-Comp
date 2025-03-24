"""
Main FastAPI application for Mistral AI inference server.
Designed for production deployment on H100 GPUs with vLLM.
"""
import logging
import os
import sys
import time
from contextlib import asynccontextmanager

import uvicorn
from fastapi import BackgroundTasks, FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

# Import application components
from app.api.endpoints import (
    chat_completions, completions, get_models, health
)
from app.core.engine import engine
from app.monitoring.telemetry import telemetry
from app.schemas.api import (
    ChatCompletionRequest, CompletionRequest
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("mistral-inference-server")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Lifecycle management for the FastAPI app.
    Initializes components on startup and cleans up on shutdown.
    """
    # Initialize monitoring
    if os.getenv("ENABLE_MONITORING", "true").lower() == "true":
        telemetry.setup()
        telemetry.instrument_fastapi(app)
        logger.info("SigNoz monitoring initialized")
    
    # Initialize the LLM engine
    if not engine.initialize():
        logger.error("Failed to initialize LLM engine")
        sys.exit(1)
    
    logger.info(f"Mistral AI inference server running with model: {engine.model_id}")
    logger.info(f"Using tensor parallelism across {engine.tensor_parallel_size} GPUs")
    
    yield
    
    # Cleanup resources
    logger.info("Shutting down Mistral AI inference server")


# Create FastAPI app
app = FastAPI(
    title="Mistral AI Inference Server",
    description="""
    # OpenAI-compatible API for Mistral models using vLLM
    
    This API provides a high-performance, OpenAI-compatible interface for Mistral models,
    optimized for production deployment on multiple H100 GPUs using vLLM.
    
    ## API Endpoints
    - `/v1/chat/completions`: Chat completions API (similar to OpenAI's ChatGPT API)
    - `/v1/completions`: Text completions API
    - `/v1/models`: List available models
    - `/health`: Service health check
    """,
    version="1.0.0",
    lifespan=lifespan,
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Add middleware for request timing and logging
@app.middleware("http")
async def add_timing_middleware(request: Request, call_next):
    """Middleware to log request timing."""
    start_time = time.time()
    
    # Get client IP and method+path for logging
    client_ip = request.client.host
    method = request.method
    path = request.url.path
    
    logger.info(f"Request received: {method} {path} from {client_ip}")
    
    try:
        response = await call_next(request)
        
        # Calculate and log processing time
        process_time = (time.time() - start_time) * 1000
        response.headers["X-Process-Time"] = f"{process_time:.2f}ms"
        logger.info(f"Request completed: {method} {path} - {response.status_code} - {process_time:.2f}ms")
        
        return response
    except Exception as e:
        logger.error(f"Request failed: {method} {path} - {str(e)}")
        return JSONResponse(
            status_code=500,
            content={"error": {"message": str(e), "type": "server_error"}}
        )


# Register API routes
app.add_api_route("/health", health, methods=["GET"])
app.add_api_route("/v1/models", get_models, methods=["GET"])
app.add_api_route("/v1/chat/completions", chat_completions, methods=["POST"])
app.add_api_route("/v1/completions", completions, methods=["POST"])


if __name__ == "__main__":
    # Get port from environment or use default
    port = int(os.getenv("PORT", "8000"))
    host = os.getenv("HOST", "0.0.0.0")
    
    # Start server
    logger.info(f"Starting Mistral AI inference server at http://{host}:{port}")
    uvicorn.run(
        "app.main:app",
        host=host,
        port=port,
        log_level="info",
        workers=1,  # Single worker as we're using multiple GPUs with vLLM
    )
