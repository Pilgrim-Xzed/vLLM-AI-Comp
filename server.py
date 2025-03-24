#!/usr/bin/env python3
import asyncio
import json
import logging
import os
import sys
import time
from contextlib import asynccontextmanager
from typing import Dict, List, Optional, Union, Any, Literal

import numpy as np
import uvicorn
from fastapi import BackgroundTasks, FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, Response, StreamingResponse
from pydantic import BaseModel, Field
from dotenv import load_dotenv
import prometheus_client
from prometheus_client import Counter, Histogram, Gauge
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter

# Import vLLM components
from vllm import LLM, SamplingParams
from vllm.utils import random_uuid

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("mistral-inference-server")

# Configure OpenTelemetry for distributed tracing
if os.getenv("ENABLE_TRACING", "false").lower() == "true":
    trace.set_tracer_provider(TracerProvider())
    otlp_exporter = OTLPSpanExporter(endpoint=os.getenv("OTLP_ENDPOINT", "localhost:4317"))
    span_processor = BatchSpanProcessor(otlp_exporter)
    trace.get_tracer_provider().add_span_processor(span_processor)

tracer = trace.get_tracer("mistral-inference-server")

# Configure Prometheus metrics
REQUESTS = Counter("requests_total", "Total number of requests received", ["endpoint", "status"])
LATENCY = Histogram("request_latency_seconds", "Request latency in seconds", ["endpoint"])
TOKENS_GENERATED = Counter("tokens_generated_total", "Total number of tokens generated", ["model"])
CONCURRENT_REQUESTS = Gauge("concurrent_requests", "Number of concurrent requests")

# Global LLM instance
llm_instance = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Lifecycle management for the FastAPI app.
    Initializes the LLM instance on startup and cleans up on shutdown.
    """
    global llm_instance
    
    # Initialize the LLM instance on startup
    model_id = os.getenv("MODEL_ID", "mistralai/mistral-7b")
    tensor_parallel_size = int(os.getenv("TENSOR_PARALLEL_SIZE", "1"))
    gpu_memory_utilization = float(os.getenv("GPU_MEMORY_UTILIZATION", "0.9"))
    max_model_len = int(os.getenv("MAX_MODEL_LEN", "8192"))
    
    logger.info(f"Initializing LLM with model: {model_id}, tensor_parallel_size: {tensor_parallel_size}")
    
    # Create LLM instance
    try:
        llm_instance = LLM(
            model=model_id,
            tensor_parallel_size=tensor_parallel_size,
            gpu_memory_utilization=gpu_memory_utilization,
            max_model_len=max_model_len,
        )
        logger.info("LLM initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize LLM: {e}")
        sys.exit(1)
    
    yield
    
    # Clean up resources on shutdown
    logger.info("Shutting down and cleaning up resources")

# Define API models
class ChatCompletionRequestMessage(BaseModel):
    role: Literal["system", "user", "assistant", "function"] = "user"
    content: str
    name: Optional[str] = None

class ChatCompletionRequest(BaseModel):
    model: str = Field(..., description="ID of the model to use for inference")
    messages: List[ChatCompletionRequestMessage] = Field(..., description="A list of messages comprising the conversation so far")
    temperature: Optional[float] = Field(0.7, description="What sampling temperature to use, between 0 and 2. Higher values like 0.8 will make the output more random, while lower values like 0.2 will make it more focused and deterministic.")
    top_p: Optional[float] = Field(1.0, description="An alternative to sampling with temperature, called nucleus sampling, where the model considers the results of the tokens with top_p probability mass. So 0.1 means only the tokens comprising the top 10% probability mass are considered.")
    max_tokens: Optional[int] = Field(None, description="The maximum number of tokens to generate in the completion.")
    n: Optional[int] = Field(1, description="How many completions to generate for each prompt.")
    stream: Optional[bool] = Field(False, description="If set, partial message deltas will be sent as data-only server-sent events.")
    stop: Optional[Union[str, List[str]]] = Field(None, description="Up to 4 sequences where the API will stop generating further tokens.")
    presence_penalty: Optional[float] = Field(0.0, description="Number between -2.0 and 2.0. Positive values penalize new tokens based on whether they appear in the text so far.")
    frequency_penalty: Optional[float] = Field(0.0, description="Number between -2.0 and 2.0. Positive values penalize new tokens based on their existing frequency in the text so far.")
    user: Optional[str] = Field(None, description="A unique identifier representing your end-user.")

    class Config:
        schema_extra = {
            "example": {
                "model": "mistralai/Mistral-7B-Instruct-v0.1",
                "messages": [
                    {"role": "system", "content": "You are a helpful AI assistant."},
                    {"role": "user", "content": "Hello! Can you help me with a coding problem?"}
                ],
                "temperature": 0.7,
                "max_tokens": 500
            }
        }

class ChatCompletionResponseChoice(BaseModel):
    index: int
    message: ChatCompletionRequestMessage
    finish_reason: Optional[str] = None

class UsageInfo(BaseModel):
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int

class ChatCompletionResponse(BaseModel):
    id: str
    object: str = "chat.completion"
    created: int
    model: str
    choices: List[ChatCompletionResponseChoice]
    usage: UsageInfo

class CompletionRequest(BaseModel):
    model: str = Field(..., description="ID of the model to use for inference")
    prompt: Union[str, List[str]] = Field(..., description="The prompt(s) to generate completions for")
    temperature: Optional[float] = Field(0.7, description="What sampling temperature to use, between 0 and 2")
    top_p: Optional[float] = Field(1.0, description="An alternative to sampling with temperature, called nucleus sampling")
    max_tokens: Optional[int] = Field(None, description="The maximum number of tokens to generate in the completion")
    n: Optional[int] = Field(1, description="How many completions to generate for each prompt")
    stream: Optional[bool] = Field(False, description="Whether to stream back partial progress")
    stop: Optional[Union[str, List[str]]] = Field(None, description="Up to 4 sequences where the API will stop generating further tokens")
    presence_penalty: Optional[float] = Field(0.0, description="Penalty for token presence")
    frequency_penalty: Optional[float] = Field(0.0, description="Penalty for token frequency")
    user: Optional[str] = Field(None, description="A unique identifier for the end-user")
    
    class Config:
        schema_extra = {
            "example": {
                "model": "mistralai/Mistral-7B-Instruct-v0.1",
                "prompt": "Once upon a time",
                "temperature": 0.7,
                "max_tokens": 100
            }
        }

class CompletionResponseChoice(BaseModel):
    index: int
    text: str
    finish_reason: Optional[str] = None
    logprobs: Optional[Any] = None

class CompletionResponse(BaseModel):
    id: str
    object: str = "text_completion"
    created: int
    model: str
    choices: List[CompletionResponseChoice]
    usage: UsageInfo

class ModelCard(BaseModel):
    id: str
    object: str = "model"
    created: int
    owned_by: str
    permission: List[Dict[str, Any]] = []
    root: str
    parent: Optional[str] = None

class ModelList(BaseModel):
    object: str = "list"
    data: List[ModelCard] = []

# Create FastAPI app
app = FastAPI(
    title="Mistral Inference Server",
    description="""
    # OpenAI-compatible API for Mistral models using vLLM
    
    This API provides an OpenAI-compatible interface for Mistral models powered by vLLM.
    
    ## API Endpoints
    - `/v1/chat/completions`: Chat completions API (similar to OpenAI's ChatGPT API)
    - `/v1/completions`: Text completions API
    - `/v1/models`: List available models
    - `/health`: Server health check
    - `/metrics`: Prometheus metrics
    """,
    version="1.0.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json",
    swagger_ui_parameters={"defaultModelsExpandDepth": -1},
    openapi_tags=[
        {
            "name": "Chat",
            "description": "Chat completions API (similar to OpenAI's ChatGPT API)"
        },
        {
            "name": "Completions", 
            "description": "Text completions API"
        },
        {
            "name": "Models",
            "description": "Model management endpoints"
        },
        {
            "name": "Server Status",
            "description": "Server health and monitoring"
        },
        {
            "name": "Monitoring",
            "description": "Prometheus metrics for monitoring"
        }
    ],
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Health check endpoint
@app.get("/health", 
    tags=["Server Status"],
    summary="Check server health",
    description="Returns the health status of the API server",
    response_description="Health status")
async def health():
    return {"status": "ok"}

# Prometheus metrics endpoint
@app.get("/metrics", 
    tags=["Monitoring"],
    summary="Get Prometheus metrics",
    description="Provides Prometheus metrics for monitoring the server",
    response_description="Prometheus metrics in text format")
async def metrics():
    return Response(content=prometheus_client.generate_latest(), media_type="text/plain")

# OpenAI-compatible Models endpoint
@app.get("/v1/models", 
    tags=["Models"],
    summary="List available models",
    description="Returns a list of the available models",
    response_model=ModelList,
    response_description="List of available models")
async def get_models():
    model_id = os.getenv("MODEL_ID", "mistralai/mistral-7b")
    model_card = ModelCard(
        id=model_id,
        created=int(time.time()),
        owned_by="mistralai",
        root=model_id,
    )
    return ModelList(data=[model_card])

# Helper functions for chat
def prepare_messages_for_inference(messages: List[ChatCompletionRequestMessage]) -> str:
    """Convert chat messages to a prompt format that Mistral understands."""
    prompt = ""
    for message in messages:
        if message.role == "system":
            prompt += f"<s>[INST] {message.content} [/INST]</s>\n"
        elif message.role == "user":
            prompt += f"<s>[INST] {message.content} [/INST]</s>\n"
        elif message.role == "assistant":
            prompt += f"{message.content}\n"
    return prompt.strip()

def create_streaming_response(request_id: str, model: str, created: int, choices):
    """Create a streaming response compatible with OpenAI's format."""
    for choice in choices:
        chunk = {
            "id": request_id,
            "object": "chat.completion.chunk",
            "created": created,
            "model": model,
            "choices": [
                {
                    "index": choice["index"],
                    "delta": {"role": "assistant", "content": choice["text"]},
                    "finish_reason": choice["finish_reason"]
                }
            ]
        }
        yield f"data: {json.dumps(chunk)}\n\n"
    yield "data: [DONE]\n\n"

# OpenAI-compatible Chat Completions endpoint
@app.post("/v1/chat/completions", 
    tags=["Chat"],
    summary="Create a chat completion",
    description="Creates a completion for the chat message",
    response_model=ChatCompletionResponse,
    response_description="The generated chat completion")
async def chat_completions(request: ChatCompletionRequest, raw_request: Request, background_tasks: BackgroundTasks):
    global llm_instance
    
    start_time = time.time()
    CONCURRENT_REQUESTS.inc()
    
    with tracer.start_as_current_span("chat_completions"):
        try:
            # Process the request
            prompt = prepare_messages_for_inference(request.messages)
            request_id = f"chatcmpl-{random_uuid()}"
            created = int(time.time())
            
            # Set up sampling parameters
            sampling_params = SamplingParams(
                temperature=request.temperature,
                top_p=request.top_p,
                max_tokens=request.max_tokens if request.max_tokens is not None else 1024,
                n=request.n,
                stop=request.stop,
                presence_penalty=request.presence_penalty,
                frequency_penalty=request.frequency_penalty,
            )
            
            # Handle streaming response
            if request.stream:
                async def generate_stream():
                    async for output in llm_instance.generate_async(
                        [prompt],
                        sampling_params,
                        request_id,
                    ):
                        prompt_tokens = len(output.prompt_token_ids)
                        generated_text = output.outputs[0].text
                        finish_reason = output.outputs[0].finish_reason
                        
                        # Increment token counters
                        TOKENS_GENERATED.labels(model=request.model).inc(len(output.outputs[0].token_ids))
                        
                        yield {
                            "index": 0,
                            "text": generated_text,
                            "finish_reason": finish_reason
                        }
                
                # Return streaming response
                background_tasks.add_task(lambda: CONCURRENT_REQUESTS.dec())
                background_tasks.add_task(
                    lambda: REQUESTS.labels(endpoint="/v1/chat/completions", status="success").inc()
                )
                background_tasks.add_task(
                    lambda: LATENCY.labels(endpoint="/v1/chat/completions").observe(time.time() - start_time)
                )
                
                return StreamingResponse(
                    create_streaming_response(request_id, request.model, created, generate_stream()),
                    media_type="text/event-stream",
                )
            
            # Handle non-streaming response
            outputs = await llm_instance.generate_async(
                [prompt],
                sampling_params,
                request_id,
            )
            output = outputs[0]
            
            prompt_tokens = len(output.prompt_token_ids)
            completion_tokens = sum(len(output.outputs[i].token_ids) for i in range(len(output.outputs)))
            TOKENS_GENERATED.labels(model=request.model).inc(completion_tokens)
            
            # Create response
            choices = []
            for i in range(len(output.outputs)):
                choices.append(
                    ChatCompletionResponseChoice(
                        index=i,
                        message=ChatCompletionRequestMessage(
                            role="assistant",
                            content=output.outputs[i].text,
                        ),
                        finish_reason=output.outputs[i].finish_reason,
                    )
                )
            
            response = ChatCompletionResponse(
                id=request_id,
                created=created,
                model=request.model,
                choices=choices,
                usage=UsageInfo(
                    prompt_tokens=prompt_tokens,
                    completion_tokens=completion_tokens,
                    total_tokens=prompt_tokens + completion_tokens,
                ),
            )
            
            REQUESTS.labels(endpoint="/v1/chat/completions", status="success").inc()
            LATENCY.labels(endpoint="/v1/chat/completions").observe(time.time() - start_time)
            CONCURRENT_REQUESTS.dec()
            
            return response
            
        except Exception as e:
            REQUESTS.labels(endpoint="/v1/chat/completions", status="error").inc()
            LATENCY.labels(endpoint="/v1/chat/completions").observe(time.time() - start_time)
            CONCURRENT_REQUESTS.dec()
            logger.error(f"Error in chat completions: {e}")
            raise HTTPException(status_code=500, detail=str(e))

# OpenAI-compatible Completions endpoint
@app.post("/v1/completions", 
    tags=["Completions"],
    summary="Create a completion",
    description="Creates a completion for the provided prompt",
    response_model=CompletionResponse,
    response_description="The generated completion")
async def completions(request: CompletionRequest, raw_request: Request, background_tasks: BackgroundTasks):
    global llm_instance
    
    start_time = time.time()
    CONCURRENT_REQUESTS.inc()
    
    with tracer.start_as_current_span("completions"):
        try:
            # Process the request
            if isinstance(request.prompt, list):
                # For simplicity, just use the first prompt in the list
                prompt = request.prompt[0] if request.prompt else ""
            else:
                prompt = request.prompt
            
            request_id = f"cmpl-{random_uuid()}"
            created = int(time.time())
            
            # Set up sampling parameters
            sampling_params = SamplingParams(
                temperature=request.temperature,
                top_p=request.top_p,
                max_tokens=request.max_tokens if request.max_tokens is not None else 1024,
                n=request.n,
                stop=request.stop,
                presence_penalty=request.presence_penalty,
                frequency_penalty=request.frequency_penalty,
            )
            
            # Handle streaming response
            if request.stream:
                async def generate_stream():
                    async for output in llm_instance.generate_async(
                        [prompt],
                        sampling_params,
                        request_id,
                    ):
                        prompt_tokens = len(output.prompt_token_ids)
                        generated_text = output.outputs[0].text
                        finish_reason = output.outputs[0].finish_reason
                        
                        # Increment token counters
                        TOKENS_GENERATED.labels(model=request.model).inc(len(output.outputs[0].token_ids))
                        
                        yield {
                            "id": request_id,
                            "object": "text_completion",
                            "created": created,
                            "model": request.model,
                            "choices": [
                                {
                                    "text": generated_text,
                                    "index": 0,
                                    "logprobs": None,
                                    "finish_reason": finish_reason,
                                }
                            ]
                        }
                
                # Return streaming response
                background_tasks.add_task(lambda: CONCURRENT_REQUESTS.dec())
                background_tasks.add_task(
                    lambda: REQUESTS.labels(endpoint="/v1/completions", status="success").inc()
                )
                background_tasks.add_task(
                    lambda: LATENCY.labels(endpoint="/v1/completions").observe(time.time() - start_time)
                )
                
                async def generate():
                    async for response in generate_stream():
                        yield f"data: {json.dumps(response)}\n\n"
                    yield "data: [DONE]\n\n"
                
                return StreamingResponse(
                    generate(),
                    media_type="text/event-stream",
                )
            
            # Handle non-streaming response
            outputs = await llm_instance.generate_async(
                [prompt],
                sampling_params,
                request_id,
            )
            output = outputs[0]
            
            prompt_tokens = len(output.prompt_token_ids)
            completion_tokens = sum(len(output.outputs[i].token_ids) for i in range(len(output.outputs)))
            TOKENS_GENERATED.labels(model=request.model).inc(completion_tokens)
            
            # Create response
            choices = []
            for i in range(len(output.outputs)):
                choices.append(
                    CompletionResponseChoice(
                        text=output.outputs[i].text,
                        index=i,
                        logprobs=None,
                        finish_reason=output.outputs[i].finish_reason,
                    )
                )
            
            response = CompletionResponse(
                id=request_id,
                created=created,
                model=request.model,
                choices=choices,
                usage=UsageInfo(
                    prompt_tokens=prompt_tokens,
                    completion_tokens=completion_tokens,
                    total_tokens=prompt_tokens + completion_tokens,
                ),
            )
            
            REQUESTS.labels(endpoint="/v1/completions", status="success").inc()
            LATENCY.labels(endpoint="/v1/completions").observe(time.time() - start_time)
            CONCURRENT_REQUESTS.dec()
            
            return response
            
        except Exception as e:
            REQUESTS.labels(endpoint="/v1/completions", status="error").inc()
            LATENCY.labels(endpoint="/v1/completions").observe(time.time() - start_time)
            CONCURRENT_REQUESTS.dec()
            logger.error(f"Error in completions: {e}")
            raise HTTPException(status_code=500, detail=str(e))

# Main function to run the server
if __name__ == "__main__":
    port = int(os.getenv("PORT", "8000"))
    host = os.getenv("HOST", "0.0.0.0")
    
    # Run the server
    uvicorn.run(
        "server:app",
        host=host,
        port=port,
        log_level="info",
        reload=False,
    )
