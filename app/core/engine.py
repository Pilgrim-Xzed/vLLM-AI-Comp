"""
Engine module for vLLM model initialization and management.
Optimized for 2xH100 GPUs in production environments.
"""
import logging
import os
from typing import Optional

from vllm import LLM, SamplingParams
from vllm.utils import random_uuid

logger = logging.getLogger("mistral-inference-engine")


class ModelEngine:
    """
    Handles vLLM model initialization and inference.
    Configured for optimal performance on multi-GPU setups.
    """

    def __init__(self):
        self.llm: Optional[LLM] = None
        self.model_id = os.getenv("MODEL_ID", "mistralai/Mistral-7B-Instruct-v0.1")
        # Use 2 for 2xH100 GPUs tensor parallelism
        self.tensor_parallel_size = int(os.getenv("TENSOR_PARALLEL_SIZE", "2"))
        # Optimal GPU memory utilization for H100s
        self.gpu_memory_utilization = float(os.getenv("GPU_MEMORY_UTILIZATION", "0.90"))
        self.max_model_len = int(os.getenv("MAX_MODEL_LEN", "8192"))
        # Use bfloat16 for optimal performance on H100 GPUs
        self.dtype = os.getenv("DTYPE", "bfloat16")
        # Additional vLLM parameters for production
        self.trust_remote_code = os.getenv("TRUST_REMOTE_CODE", "true").lower() == "true"
        self.enforce_eager = os.getenv("ENFORCE_EAGER", "false").lower() == "true"
        # For multi-GPU scheduling
        self.max_parallel_requests = int(os.getenv("MAX_PARALLEL_REQUESTS", "128"))

    def initialize(self) -> bool:
        """
        Initialize the LLM engine with tensor parallelism and optimizations.
        Returns True if successful, False otherwise.
        """
        try:
            logger.info(
                f"Initializing LLM engine with model: {self.model_id}, "
                f"tensor_parallel_size: {self.tensor_parallel_size}, "
                f"gpu_memory_utilization: {self.gpu_memory_utilization}, "
                f"dtype: {self.dtype}"
            )
            
            self.llm = LLM(
                model=self.model_id,
                tensor_parallel_size=self.tensor_parallel_size,
                gpu_memory_utilization=self.gpu_memory_utilization,
                max_model_len=self.max_model_len,
                dtype=self.dtype,
                trust_remote_code=self.trust_remote_code,
                enforce_eager=self.enforce_eager,
            )
            
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

    def generate(self, prompt, sampling_params) -> dict:
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
            "dtype": self.dtype,
        }
