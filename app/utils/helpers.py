"""
Helper utilities for the vLLM inference server.
"""
import logging
import os
import uuid
from typing import Optional, Union, List

# Setup logging
logger = logging.getLogger(__name__)

def random_uuid_generator() -> str:
    """Generate a random UUID string."""
    return str(uuid.uuid4())

def get_token_count(text: Union[str, List[str]]) -> int:
    """
    Estimate the number of tokens in a text string.
    This is a simple approximation - for more accurate counts,
    use the model's tokenizer directly.
    """
    if isinstance(text, list):
        text = " ".join(text)
    
    # Simple approximation - 1 token is roughly 4 characters for English text
    # For production, use the actual tokenizer
    return len(text) // 4

def get_env_bool(name: str, default: bool = False) -> bool:
    """Get a boolean environment variable."""
    value = os.getenv(name, str(default)).lower()
    return value in ("true", "1", "yes", "y", "t")

def get_env_int(name: str, default: int) -> int:
    """Get an integer environment variable."""
    try:
        return int(os.getenv(name, str(default)))
    except ValueError:
        logger.warning(f"Invalid value for {name}, using default: {default}")
        return default

def get_env_float(name: str, default: float) -> float:
    """Get a float environment variable."""
    try:
        return float(os.getenv(name, str(default)))
    except ValueError:
        logger.warning(f"Invalid value for {name}, using default: {default}")
        return default

def format_error_response(error: Exception, status_code: int = 500) -> dict:
    """Format a standard error response."""
    error_type = type(error).__name__
    return {
        "error": {
            "message": str(error),
            "type": error_type,
            "code": status_code
        }
    }

def format_chat_prompt(system: Optional[str], user: str) -> str:
    """
    Format a prompt for Mistral's chat format.
    """
    if system:
        return f"<s>[INST] {system}\n\n{user} [/INST]"
    else:
        return f"<s>[INST] {user} [/INST]"
