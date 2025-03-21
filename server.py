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
TOKENS_GENERATED = Counter("tokens_generated_total", "Total number of tokens generated")
TOKENS_INPUT = Counter("tokens_input_total", "Total number of input tokens processed")
GPU_MEMORY_ALLOCATED = Gauge("gpu_memory_allocated_bytes", "GPU memory allocated in bytes", ["device"])
MODEL_LOAD_TIME = Histogram("model_load_time_seconds", "Time to load the model in seconds")

# Initialize LLM
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load the model on startup
    global llm
    
    model_load_start = time.time()
    
    model_id = os.getenv("MODEL_ID", "mistralai/mistral-7b")
    tensor_parallel_size = int(os.getenv("TENSOR_PARALLEL_SIZE", "2"))  # 2 for 2xH100
    gpu_memory_utilization = float(os.getenv("GPU_MEMORY_UTILIZATION", "0.9"))
    max_model_len = int(os.getenv("MAX_MODEL_LEN", "8192"))
    
    logger.info(f"Loading model {model_id} with tensor parallelism {tensor_parallel_size}")
    
    llm = LLM(
        model=model_id,
        tensor_parallel_size=tensor_parallel_size,
        gpu_memory_utilization=gpu_memory_utilization,
        max_model_len=max_model_len,
        trust_remote_code=True,
        enforce_eager=False,
    )
    
    model_load_end = time.time()
    MODEL_LOAD_TIME.observe(model_load_end - model_load_start)
    
    logger.info(f"Model loaded in {model_load_end - model_load_start:.2f} seconds")
    
    yield
    
    # Clean up resources on shutdown
    logger.info("Shutting down and cleaning up resources")

# Define API models
class ChatCompletionRequestMessage(BaseModel):
    role: Literal["system", "user", "assistant", "function"] = "user"
    content: str
    name: Optional[str] = None

class ChatCompletionRequest(BaseModel):
    model: str
    messages: List[ChatCompletionRequestMessage]
    temperature: Optional[float] = 0.7
    top_p: Optional[float] = 1.0
    max_tokens: Optional[int] = None
    n: Optional[int] = 1
    stream: Optional[bool] = False
    stop: Optional[Union[str, List[str]]] = None
    presence_penalty: Optional[float] = 0.0
    frequency_penalty: Optional[float] = 0.0
    user: Optional[str] = None

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
    model: str
    prompt: Union[str, List[str]]
    temperature: Optional[float] = 0.7
    top_p: Optional[float] = 1.0
    max_tokens: Optional[int] = None
    n: Optional[int] = 1
    stream: Optional[bool] = False
    stop: Optional[Union[str, List[str]]] = None
    presence_penalty: Optional[float] = 0.0
    frequency_penalty: Optional[float] = 0.0
    user: Optional[str] = None

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
    description="OpenAI-compatible API for Mistral models using vLLM",
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

# Health check endpoint
@app.get("/health")
async def health():
    return {"status": "ok"}

# Prometheus metrics endpoint
@app.get("/metrics")
async def metrics():
    return Response(content=prometheus_client.generate_latest(), media_type="text/plain")

# OpenAI-compatible Models endpoint
@app.get("/v1/models")
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
            prompt += f"<s>[INST] {message.content} [/INST]"
        elif message.role == "assistant":
            prompt += f" {message.content}</s>\n"
    
    # If the last message is from the user, we need to add the assistant prefix
    if messages[-1].role == "user":
        prompt += " "
    
    return prompt

def create_streaming_response(request_id: str, model: str, created: int, choices):
    for choice in choices:
        data = {
            "id": request_id,
            "object": "chat.completion.chunk",
            "created": created,
            "model": model,
            "choices": [choice],
        }
        yield f"data: {json.dumps(data)}\n\n"
    yield "data: [DONE]\n\n"

# OpenAI-compatible Chat Completions endpoint
@app.post("/v1/chat/completions")
async def chat_completions(request: ChatCompletionRequest, raw_request: Request, background_tasks: BackgroundTasks):
    with tracer.start_as_current_span("chat_completions"):
        request_start = time.time()
        request_id = f"chatcmpl-{random_uuid()}"
        created = int(time.time())
        
        try:
            prompt = prepare_messages_for_inference(request.messages)
            
            sampling_params = SamplingParams(
                temperature=request.temperature,
                top_p=request.top_p,
                max_tokens=request.max_tokens if request.max_tokens is not None else 4096,
                stop=request.stop if request.stop else None,
                presence_penalty=request.presence_penalty,
                frequency_penalty=request.frequency_penalty,
            )
            
            input_token_count = llm.get_tokenizer().encode(prompt, add_special_tokens=False).__len__()
            TOKENS_INPUT.inc(input_token_count)
            
            # Process completion request
            if request.stream:
                # Streaming response
                async def generate():
                    async for output in llm.generate_stream(prompt, sampling_params):
                        if output.outputs[0].text:
                            chunk = {
                                "index": 0,
                                "delta": {"role": "assistant", "content": output.outputs[0].text},
                                "finish_reason": None,
                            }
                            yield f"data: {json.dumps(chunk)}\n\n"
                            
                    # Final chunk with finish_reason
                    final_chunk = {
                        "index": 0,
                        "delta": {},
                        "finish_reason": "stop",
                    }
                    yield f"data: {json.dumps(final_chunk)}\n\n"
                    yield "data: [DONE]\n\n"
                
                background_tasks.add_task(lambda: REQUESTS.labels(endpoint="/v1/chat/completions", status="success").inc())
                request_end = time.time()
                background_tasks.add_task(lambda: LATENCY.labels(endpoint="/v1/chat/completions").observe(request_end - request_start))
                
                return StreamingResponse(generate(), media_type="text/event-stream")
            else:
                # Non-streaming response
                results = await llm.generate(prompt, sampling_params)
                result = results[0]
                
                completion_token_count = len(result.outputs[0].token_ids)
                TOKENS_GENERATED.inc(completion_token_count)
                
                choices = [{
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": result.outputs[0].text,
                    },
                    "finish_reason": "stop" if result.outputs[0].finish_reason == "stop" else None,
                }]
                
                response = ChatCompletionResponse(
                    id=request_id,
                    created=created,
                    model=request.model,
                    choices=choices,
                    usage=UsageInfo(
                        prompt_tokens=input_token_count,
                        completion_tokens=completion_token_count,
                        total_tokens=input_token_count + completion_token_count,
                    ),
                )
                
                REQUESTS.labels(endpoint="/v1/chat/completions", status="success").inc()
                request_end = time.time()
                LATENCY.labels(endpoint="/v1/chat/completions").observe(request_end - request_start)
                
                return response
                
        except Exception as e:
            logger.error(f"Error in chat completions: {str(e)}")
            REQUESTS.labels(endpoint="/v1/chat/completions", status="error").inc()
            raise HTTPException(status_code=500, detail=str(e))

# OpenAI-compatible Completions endpoint
@app.post("/v1/completions")
async def completions(request: CompletionRequest, raw_request: Request, background_tasks: BackgroundTasks):
    with tracer.start_as_current_span("completions"):
        request_start = time.time()
        request_id = f"cmpl-{random_uuid()}"
        created = int(time.time())
        
        try:
            prompt = request.prompt if isinstance(request.prompt, str) else request.prompt[0]
            
            sampling_params = SamplingParams(
                temperature=request.temperature,
                top_p=request.top_p,
                max_tokens=request.max_tokens if request.max_tokens is not None else 4096,
                stop=request.stop if request.stop else None,
                presence_penalty=request.presence_penalty,
                frequency_penalty=request.frequency_penalty,
            )
            
            input_token_count = llm.get_tokenizer().encode(prompt, add_special_tokens=False).__len__()
            TOKENS_INPUT.inc(input_token_count)
            
            # Process completion request
            if request.stream:
                # Streaming response
                async def generate():
                    async for output in llm.generate_stream(prompt, sampling_params):
                        if output.outputs[0].text:
                            chunk = {
                                "id": request_id,
                                "object": "text_completion.chunk",
                                "created": created,
                                "model": request.model,
                                "choices": [{
                                    "index": 0,
                                    "text": output.outputs[0].text,
                                    "finish_reason": None,
                                }],
                            }
                            yield f"data: {json.dumps(chunk)}\n\n"
                            
                    # Final chunk with finish_reason
                    final_chunk = {
                        "id": request_id,
                        "object": "text_completion.chunk",
                        "created": created,
                        "model": request.model,
                        "choices": [{
                            "index": 0,
                            "text": "",
                            "finish_reason": "stop",
                        }],
                    }
                    yield f"data: {json.dumps(final_chunk)}\n\n"
                    yield "data: [DONE]\n\n"
                
                background_tasks.add_task(lambda: REQUESTS.labels(endpoint="/v1/completions", status="success").inc())
                request_end = time.time()
                background_tasks.add_task(lambda: LATENCY.labels(endpoint="/v1/completions").observe(request_end - request_start))
                
                return StreamingResponse(generate(), media_type="text/event-stream")
            else:
                # Non-streaming response
                results = await llm.generate(prompt, sampling_params)
                result = results[0]
                
                completion_token_count = len(result.outputs[0].token_ids)
                TOKENS_GENERATED.inc(completion_token_count)
                
                choices = [{
                    "index": 0,
                    "text": result.outputs[0].text,
                    "finish_reason": "stop" if result.outputs[0].finish_reason == "stop" else None,
                    "logprobs": None,
                }]
                
                response = CompletionResponse(
                    id=request_id,
                    created=created,
                    model=request.model,
                    choices=choices,
                    usage=UsageInfo(
                        prompt_tokens=input_token_count,
                        completion_tokens=completion_token_count,
                        total_tokens=input_token_count + completion_token_count,
                    ),
                )
                
                REQUESTS.labels(endpoint="/v1/completions", status="success").inc()
                request_end = time.time()
                LATENCY.labels(endpoint="/v1/completions").observe(request_end - request_start)
                
                return response
                
        except Exception as e:
            logger.error(f"Error in completions: {str(e)}")
            REQUESTS.labels(endpoint="/v1/completions", status="error").inc()
            raise HTTPException(status_code=500, detail=str(e))

# Main function to run the server
if __name__ == "__main__":
    port = int(os.getenv("PORT", "8000"))
    host = os.getenv("HOST", "0.0.0.0")
    
    logger.info(f"Starting server on {host}:{port}")
    
    uvicorn.run(
        "server:app",
        host=host,
        port=port,
        log_level="info",
        workers=1,  # We use a single worker as vLLM handles the parallelism
    )
