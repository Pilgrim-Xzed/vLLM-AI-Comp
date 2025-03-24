"""
API endpoint implementations for the vLLM service.
OpenAI-compatible endpoints for chat and text completions.
"""
import asyncio
import json
import logging
import time
from typing import List, Optional, Union

from fastapi import BackgroundTasks, Request
from fastapi.responses import JSONResponse, StreamingResponse

from app.core.engine import ModelEngine
from app.schemas.api import (
    ChatCompletionRequest, ChatCompletionRequestMessage, ChatCompletionResponse,
    ChatCompletionResponseChoice, CompletionRequest, CompletionResponse,
    CompletionResponseChoice, ModelCard, ModelList, UsageInfo
)
from app.monitoring.telemetry import telemetry
from app.utils.helpers import get_token_count, random_uuid_generator

logger = logging.getLogger("mistral-inference-api")

# Global model engine instance
engine = ModelEngine()


# Health endpoint
async def health():
    """Health check endpoint."""
    if not engine.llm:
        return JSONResponse(
            status_code=503,
            content={"status": "unavailable", "message": "Model engine not initialized"}
        )
    return JSONResponse(
        status_code=200,
        content={"status": "ok", "message": "Service is healthy"}
    )


# Model information endpoint
async def get_models():
    """Get information about available models."""
    with telemetry.tracer.start_as_current_span("get_models"):
        telemetry.track_request("get_models")
        start_time = time.time()
        
        try:
            model_info = engine.get_model_info()
            timestamp = int(time.time())
            
            model = ModelCard(
                id=model_info["id"],
                object="model",
                created=timestamp,
                owned_by="mistral-ai",
                root=model_info["id"],
                permission=[]
            )
            
            response = ModelList(
                object="list",
                data=[model]
            )
            
            telemetry.track_latency("get_models", time.time() - start_time)
            return JSONResponse(content=response.dict())
        except Exception as e:
            logger.error(f"Error getting models: {str(e)}")
            telemetry.track_request("get_models", "error")
            return JSONResponse(
                status_code=500,
                content={"error": str(e)}
            )


# Helper for chat API
def prepare_messages_for_inference(messages: List[ChatCompletionRequestMessage]) -> str:
    """
    Convert chat messages to a prompt format that Mistral understands.
    
    This implements the chat template used by Mistral Instruct models:
    <s>[INST] System prompt [/INST]</s>
    <s>[INST] User message [/INST] Assistant message</s>
    <s>[INST] User message [/INST]
    """
    prompt = ""
    system_message = None
    
    # Extract system message if present
    for i, message in enumerate(messages):
        if message.role == "system":
            system_message = message.content
            break
    
    # Process conversation messages
    for i, message in enumerate(messages):
        if message.role == "system":
            continue
        
        if message.role == "user":
            # If this is the first user message and we have a system message,
            # prepend the system message
            if system_message and not prompt:
                prompt += f"<s>[INST] {system_message} "
                prompt += f"{message.content} [/INST]"
            else:
                prompt += f"<s>[INST] {message.content} [/INST]"
        elif message.role == "assistant":
            prompt += f" {message.content}</s>"
            
    return prompt


# Helper for streaming responses
async def create_streaming_response(request_id: str, model: str, created: int, choices):
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
                    "delta": {
                        "role": "assistant",
                        "content": choice["text"]
                    },
                    "finish_reason": choice.get("finish_reason")
                }
            ]
        }
        yield f"data: {json.dumps(chunk)}\n\n"
        
        # If we have a finish reason, we're done
        if choice.get("finish_reason"):
            yield "data: [DONE]\n\n"


@telemetry.trace_function("chat_completions")
async def chat_completions(request: ChatCompletionRequest, raw_request: Request, background_tasks: BackgroundTasks):
    """
    OpenAI-compatible chat completions endpoint.
    """
    start_time = time.time()
    telemetry.track_request("chat_completions")
    telemetry.track_concurrent(1)
    
    try:
        # Prepare prompt from chat messages
        prompt = prepare_messages_for_inference(request.messages)
        request_id = random_uuid_generator()
        
        # Create sampling parameters
        sampling_params = engine.create_sampling_params(
            temperature=request.temperature,
            top_p=request.top_p,
            max_tokens=request.max_tokens if request.max_tokens else 4096,
            n=request.n,
            stop=request.stop,
            presence_penalty=request.presence_penalty,
            frequency_penalty=request.frequency_penalty,
        )
        
        # Start the timer for generation
        generation_start = time.time()
        
        # Handle streaming response
        if request.stream:
            async def generate_stream():
                try:
                    # Generate streaming response
                    outputs = engine.llm.generate(prompt, sampling_params, request_id=request_id)
                    created = int(time.time())
                    
                    for output in outputs:
                        for i, generated_text in enumerate(output.outputs):
                            choices = [{
                                "index": i,
                                "text": generated_text.text,
                                "finish_reason": generated_text.finish_reason,
                            }]
                            async for chunk in create_streaming_response(request_id, request.model, created, choices):
                                yield chunk
                                
                    # Track token count for telemetry
                    prompt_tokens = get_token_count(prompt)
                    completion_tokens = sum(len(output.outputs[0].token_ids) for output in outputs)
                    telemetry.track_tokens(request.model, completion_tokens)
                    
                except Exception as e:
                    logger.error(f"Error in streaming generation: {str(e)}")
                    error_response = {
                        "error": {
                            "message": str(e),
                            "type": "server_error",
                            "code": 500
                        }
                    }
                    yield f"data: {json.dumps(error_response)}\n\n"
                finally:
                    telemetry.track_concurrent(-1)
                    telemetry.track_latency("chat_completions", time.time() - start_time)
            
            return StreamingResponse(
                generate_stream(),
                media_type="text/event-stream"
            )
        
        # Handle non-streaming response
        else:
            # Generate response
            outputs = engine.llm.generate(prompt, sampling_params, request_id=request_id)
            
            # Calculate token usage
            prompt_tokens = get_token_count(prompt)
            completion_tokens = sum(len(output.outputs[0].token_ids) for output in outputs)
            total_tokens = prompt_tokens + completion_tokens
            
            # Track token usage
            telemetry.track_tokens(request.model, completion_tokens)
            
            # Prepare response
            choices = []
            for i, output in enumerate(outputs):
                if i >= request.n:
                    break
                    
                choices.append(
                    ChatCompletionResponseChoice(
                        index=i,
                        message=ChatCompletionRequestMessage(
                            role="assistant",
                            content=output.outputs[0].text
                        ),
                        finish_reason=output.outputs[0].finish_reason
                    )
                )
            
            # Create response object
            response = ChatCompletionResponse(
                id=request_id,
                object="chat.completion",
                created=int(time.time()),
                model=request.model,
                choices=choices,
                usage=UsageInfo(
                    prompt_tokens=prompt_tokens,
                    completion_tokens=completion_tokens,
                    total_tokens=total_tokens
                )
            )
            
            # Log generation time
            generation_time = time.time() - generation_start
            logger.info(f"Generation completed in {generation_time:.2f}s for {completion_tokens} tokens")
            
            # Track telemetry
            telemetry.track_latency("chat_completions", time.time() - start_time)
            telemetry.track_concurrent(-1)
            
            return JSONResponse(content=response.dict())
    
    except Exception as e:
        telemetry.track_request("chat_completions", "error")
        telemetry.track_concurrent(-1)
        logger.error(f"Error in chat completions: {str(e)}")
        return JSONResponse(
            status_code=500,
            content={"error": {"message": str(e), "type": "server_error", "code": 500}}
        )


@telemetry.trace_function("completions")
async def completions(request: CompletionRequest, raw_request: Request, background_tasks: BackgroundTasks):
    """
    OpenAI-compatible completions endpoint.
    """
    start_time = time.time()
    telemetry.track_request("completions")
    telemetry.track_concurrent(1)
    
    try:
        # Handle array of prompts
        if isinstance(request.prompt, list):
            prompt = request.prompt[0]
        else:
            prompt = request.prompt
            
        request_id = random_uuid_generator()
        
        # Create sampling parameters
        sampling_params = engine.create_sampling_params(
            temperature=request.temperature,
            top_p=request.top_p,
            max_tokens=request.max_tokens if request.max_tokens else 4096,
            n=request.n,
            stop=request.stop,
            presence_penalty=request.presence_penalty,
            frequency_penalty=request.frequency_penalty,
        )
        
        # Start the timer for generation
        generation_start = time.time()
        
        # Handle streaming response
        if request.stream:
            async def generate_stream():
                try:
                    # Generate streaming response
                    outputs = engine.llm.generate(prompt, sampling_params, request_id=request_id)
                    created = int(time.time())
                    
                    for output in outputs:
                        for i, generated_text in enumerate(output.outputs):
                            chunk = {
                                "id": request_id,
                                "object": "text_completion.chunk",
                                "created": created,
                                "model": request.model,
                                "choices": [{
                                    "index": i,
                                    "text": generated_text.text,
                                    "finish_reason": generated_text.finish_reason,
                                }]
                            }
                            yield f"data: {json.dumps(chunk)}\n\n"
                            
                            if generated_text.finish_reason:
                                yield "data: [DONE]\n\n"
                                
                    # Track token count for telemetry
                    prompt_tokens = get_token_count(prompt)
                    completion_tokens = sum(len(output.outputs[0].token_ids) for output in outputs)
                    telemetry.track_tokens(request.model, completion_tokens)
                    
                except Exception as e:
                    logger.error(f"Error in streaming generation: {str(e)}")
                    error_response = {
                        "error": {
                            "message": str(e),
                            "type": "server_error",
                            "code": 500
                        }
                    }
                    yield f"data: {json.dumps(error_response)}\n\n"
                finally:
                    telemetry.track_concurrent(-1)
                    telemetry.track_latency("completions", time.time() - start_time)
            
            return StreamingResponse(
                generate_stream(),
                media_type="text/event-stream"
            )
        
        # Handle non-streaming response
        else:
            # Generate response
            outputs = engine.llm.generate(prompt, sampling_params, request_id=request_id)
            
            # Calculate token usage
            prompt_tokens = get_token_count(prompt)
            completion_tokens = sum(len(output.outputs[0].token_ids) for output in outputs)
            total_tokens = prompt_tokens + completion_tokens
            
            # Track token usage
            telemetry.track_tokens(request.model, completion_tokens)
            
            # Prepare response
            choices = []
            for i, output in enumerate(outputs):
                if i >= request.n:
                    break
                    
                choices.append(
                    CompletionResponseChoice(
                        index=i,
                        text=output.outputs[0].text,
                        finish_reason=output.outputs[0].finish_reason,
                        logprobs=None  # Not implemented in this version
                    )
                )
            
            # Create response object
            response = CompletionResponse(
                id=request_id,
                object="text_completion",
                created=int(time.time()),
                model=request.model,
                choices=choices,
                usage=UsageInfo(
                    prompt_tokens=prompt_tokens,
                    completion_tokens=completion_tokens,
                    total_tokens=total_tokens
                )
            )
            
            # Log generation time
            generation_time = time.time() - generation_start
            logger.info(f"Generation completed in {generation_time:.2f}s for {completion_tokens} tokens")
            
            # Track telemetry
            telemetry.track_latency("completions", time.time() - start_time)
            telemetry.track_concurrent(-1)
            
            return JSONResponse(content=response.dict())
    
    except Exception as e:
        telemetry.track_request("completions", "error")
        telemetry.track_concurrent(-1)
        logger.error(f"Error in completions: {str(e)}")
        return JSONResponse(
            status_code=500,
            content={"error": {"message": str(e), "type": "server_error", "code": 500}}
        )
