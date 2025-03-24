"""
Schema definitions for the API endpoints.
OpenAI-compatible schemas for interoperability.
"""
from typing import Dict, List, Optional, Union, Any, Literal
from pydantic import BaseModel, Field


class ChatCompletionRequestMessage(BaseModel):
    """Single message in a chat conversation."""
    role: Literal["system", "user", "assistant", "function"] = "user"
    content: str
    name: Optional[str] = None


class ChatCompletionRequest(BaseModel):
    """Request format for chat completions API."""
    model: str = Field(..., description="ID of the model to use for inference")
    messages: List[ChatCompletionRequestMessage] = Field(..., description="A list of messages comprising the conversation so far")
    temperature: Optional[float] = Field(0.7, description="What sampling temperature to use, between 0 and 2. Higher values like 0.8 will make the output more random, while lower values like 0.2 will make it more focused and deterministic.")
    top_p: Optional[float] = Field(1.0, description="An alternative to sampling with temperature, called nucleus sampling, where the model considers the results of the tokens with top_p probability mass.")
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
                    {"role": "user", "content": "Hello! Can you help me with a problem?"}
                ],
                "temperature": 0.7,
                "max_tokens": 500
            }
        }


class ChatCompletionResponseChoice(BaseModel):
    """Single choice in a chat completion response."""
    index: int
    message: ChatCompletionRequestMessage
    finish_reason: Optional[str] = None


class UsageInfo(BaseModel):
    """Token usage information."""
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


class ChatCompletionResponse(BaseModel):
    """Response format for chat completions API."""
    id: str
    object: str = "chat.completion"
    created: int
    model: str
    choices: List[ChatCompletionResponseChoice]
    usage: UsageInfo


class CompletionRequest(BaseModel):
    """Request format for completions API."""
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
    """Single choice in a completion response."""
    index: int
    text: str
    finish_reason: Optional[str] = None
    logprobs: Optional[Any] = None


class CompletionResponse(BaseModel):
    """Response format for completions API."""
    id: str
    object: str = "text_completion"
    created: int
    model: str
    choices: List[CompletionResponseChoice]
    usage: UsageInfo


class ModelCard(BaseModel):
    """Information about a model."""
    id: str
    object: str = "model"
    created: int
    owned_by: str
    permission: List[Dict[str, Any]] = []
    root: str
    parent: Optional[str] = None


class ModelList(BaseModel):
    """List of available models."""
    object: str = "list"
    data: List[ModelCard] = []
