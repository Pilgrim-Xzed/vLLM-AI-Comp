"""
Engine module for vLLM model initialization and management.
Optimized for high-throughput inference with quantized LLMs.
"""
import logging
import os
from typing import Optional

from vllm import LLM, SamplingParams
from vllm.utils import random_uuid

logger = logging.getLogger("llama-inference-engine")


class ModelEngine:
    """
    Handles vLLM model initialization and inference.
    Configured for optimal performance with quantized Llama 8B.
    """

    def __init__(self):
        self.llm: Optional[LLM] = None
        # Default to Llama 8B
        self.model_id = os.getenv("MODEL_ID", "meta-llama/Meta-Llama-3-8B")
        # Use 1 for quantized model (better performance with 1 GPU for quantized models)
        self.tensor_parallel_size = int(os.getenv("TENSOR_PARALLEL_SIZE", "1"))
        # Optimal GPU memory utilization for quantized models
        self.gpu_memory_utilization = float(os.getenv("GPU_MEMORY_UTILIZATION", "0.95"))
        # Set maximum sequence length
        self.max_model_len = int(os.getenv("MAX_MODEL_LEN", "8192"))
        # Use quantization for optimal throughput
        self.quantization = os.getenv("QUANTIZATION", "awq")
        # Dtype option
        self.dtype = os.getenv("DTYPE", "half")
        # Additional vLLM parameters for production
        self.trust_remote_code = os.getenv("TRUST_REMOTE_CODE", "true").lower() == "true"
        self.enforce_eager = os.getenv("ENFORCE_EAGER", "false").lower() == "true"
        # For batch scheduling optimization
        self.max_parallel_requests = int(os.getenv("MAX_PARALLEL_REQUESTS", "256"))
        # KV cache optimization parameters
        self.block_size = int(os.getenv("BLOCK_SIZE", "16"))
        self.swap_space = int(os.getenv("SWAP_SPACE", "4"))
        # Disable cuda graph tracing for quantized models
        self.disable_traces = os.getenv("DISABLE_TRACES", "true").lower() == "true"
        
    def initialize(self) -> bool:
        """
        Initialize the LLM engine with quantization and optimizations.
        Returns True if successful, False otherwise.
        """
        try:
            logger.info(
                f"Initializing LLM engine with model: {self.model_id}, "
                f"quantization: {self.quantization}, "
                f"tensor_parallel_size: {self.tensor_parallel_size}, "
                f"gpu_memory_utilization: {self.gpu_memory_utilization}, "
                f"max_parallel_requests: {self.max_parallel_requests}"
            )
            
            # Prepare kwargs with base configurations
            kwargs = {
                "model": self.model_id,
                "tensor_parallel_size": self.tensor_parallel_size,
                "gpu_memory_utilization": self.gpu_memory_utilization,
                "max_model_len": self.max_model_len,
                "trust_remote_code": self.trust_remote_code,
                "enforce_eager": self.enforce_eager,
                "disable_custom_all_reduce": True,  # Better for quantized models
                "max_parallel_loading_workers": 4,  # Faster model loading
                "seed": 42,  # Ensure deterministic results
                "block_size": self.block_size,  # Optimized KV cache blocking
                "swap_space": self.swap_space,  # Disk swap for longer contexts
                "max_num_batched_tokens": 8192,  # Increase batch size for throughput
                "max_num_seqs": self.max_parallel_requests,  # Higher throughput
            }
            
            # Add quantization-specific parameters
            if self.quantization.lower() in ["awq", "gptq", "squeezellm"]:
                kwargs["quantization"] = self.quantization.lower()
                # For quantized models, use half precision and disable tracing
                kwargs["dtype"] = "half"
                kwargs["disable_traces"] = self.disable_traces
            else:
                # For non-quantized, use the configured dtype
                kwargs["dtype"] = self.dtype
            
            self.llm = LLM(**kwargs)
            
            logger.info("LLM engine initialized successfully")
            return True
        except Exception as e:
            logger.error(f"Failed to initialize LLM engine: {e}")
            return False

    def create_sampling_params(self, **kwargs) -> SamplingParams:
        """
        Create sampling parameters for text generation.
        """
        return SamplingParams(**kwargs)

    def generate(self, prompt, sampling_params, **kwargs) -> dict:
        """
        Generate text from the LLM based on the prompt and sampling parameters.
        """
        if not self.llm:
            raise RuntimeError("LLM engine not initialized")
        
        request_id = kwargs.get("request_id", random_uuid())
        return self.llm.generate(prompt, sampling_params, request_id)

    def get_model_info(self) -> dict:
        """
        Get information about the loaded model.
        """
        if not self.llm:
            raise RuntimeError("LLM engine not initialized")
            
        return {
            "id": self.model_id,
            "tensor_parallel_size": self.tensor_parallel_size,
            "max_model_len": self.max_model_len,
            "quantization": self.quantization,
            "dtype": self.dtype,
        }
