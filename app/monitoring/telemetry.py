"""
Monitoring and telemetry using SigNoz.
Provides distributed tracing, metrics, and logging for the vLLM service.
"""
import logging
import os
import time
from functools import wraps
from typing import Callable, Dict, Optional

from opentelemetry import metrics, trace
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.metrics.export import ConsoleMetricExporter, PeriodicExportingMetricReader
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
from opentelemetry.exporter.otlp.proto.grpc.metric_exporter import OTLPMetricExporter
from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor

# Configure logging
logger = logging.getLogger("mistral-inference-monitoring")

# Service name for tracing and metrics
SERVICE_NAME = os.getenv("SERVICE_NAME", "mistral-inference-server")

class Telemetry:
    """
    Telemetry and monitoring implementation using SigNoz (OpenTelemetry-based).
    """
    def __init__(self):
        self.tracer = None
        self.meter = None
        # SigNoz endpoint (typically running as a service)
        self.otlp_endpoint = os.getenv("OTLP_ENDPOINT", "http://signoz:4317")
        # Counters and gauges for metrics
        self.request_counter = None
        self.latency_histogram = None
        self.tokens_counter = None
        self.concurrent_requests = None

    def setup(self):
        """
        Set up OpenTelemetry with SigNoz exporters for traces and metrics.
        """
        # Create resource with service information
        resource = Resource.create({"service.name": SERVICE_NAME})
        
        # Set up tracing
        trace_provider = TracerProvider(resource=resource)
        otlp_trace_exporter = OTLPSpanExporter(endpoint=self.otlp_endpoint)
        trace_provider.add_span_processor(BatchSpanProcessor(otlp_trace_exporter))
        trace.set_tracer_provider(trace_provider)
        self.tracer = trace.get_tracer(SERVICE_NAME)
        
        # Set up metrics
        meter_provider = MeterProvider(resource=resource)
        otlp_metric_exporter = OTLPMetricExporter(endpoint=self.otlp_endpoint)
        metric_reader = PeriodicExportingMetricReader(
            exporter=otlp_metric_exporter,
            export_interval_millis=15000  # Export every 15 seconds
        )
        meter_provider.add_metric_reader(metric_reader)
        metrics.set_meter_provider(meter_provider)
        self.meter = metrics.get_meter(SERVICE_NAME)
        
        # Create metrics
        self.request_counter = self.meter.create_counter(
            name="inference_requests_total",
            description="Total number of inference requests",
            unit="1"
        )
        
        self.latency_histogram = self.meter.create_histogram(
            name="inference_request_duration",
            description="Inference request duration in seconds",
            unit="s"
        )
        
        self.tokens_counter = self.meter.create_counter(
            name="tokens_generated_total",
            description="Total number of tokens generated",
            unit="1"
        )
        
        self.concurrent_requests = self.meter.create_up_down_counter(
            name="concurrent_requests",
            description="Number of concurrent requests",
            unit="1"
        )
        
        logger.info(f"Telemetry initialized with SigNoz endpoint at {self.otlp_endpoint}")

    def instrument_fastapi(self, app):
        """
        Instrument FastAPI for automatic request tracing.
        """
        FastAPIInstrumentor.instrument_app(app)
        logger.info("FastAPI instrumentation enabled")

    def track_request(self, endpoint: str, status: str = "success"):
        """
        Track an API request.
        """
        if self.request_counter:
            self.request_counter.add(1, {"endpoint": endpoint, "status": status})

    def track_latency(self, endpoint: str, duration: float):
        """
        Track request latency.
        """
        if self.latency_histogram:
            self.latency_histogram.record(duration, {"endpoint": endpoint})

    def track_tokens(self, model: str, count: int):
        """
        Track number of tokens generated.
        """
        if self.tokens_counter:
            self.tokens_counter.add(count, {"model": model})

    def track_concurrent(self, delta: int = 1):
        """
        Track concurrent requests (increment/decrement).
        """
        if self.concurrent_requests:
            self.concurrent_requests.add(delta)

    def trace_function(self, name: Optional[str] = None):
        """
        Decorator to trace a function execution.
        """
        def decorator(func: Callable):
            @wraps(func)
            def wrapper(*args, **kwargs):
                if not self.tracer:
                    return func(*args, **kwargs)
                
                span_name = name or func.__name__
                with self.tracer.start_as_current_span(span_name) as span:
                    # Add function name as attribute
                    span.set_attribute("function.name", func.__name__)
                    
                    # Add kwargs as span attributes (excluding sensitive data)
                    for key, value in kwargs.items():
                        if not self._is_sensitive(key):
                            span.set_attribute(f"arg.{key}", str(value)[:100])
                    
                    start_time = time.time()
                    try:
                        result = func(*args, **kwargs)
                        span.set_status(trace.Status(trace.StatusCode.OK))
                        return result
                    except Exception as e:
                        span.set_status(trace.Status(trace.StatusCode.ERROR, str(e)))
                        span.record_exception(e)
                        raise
                    finally:
                        duration = time.time() - start_time
                        if self.latency_histogram:
                            self.latency_histogram.record(
                                duration, 
                                {"function": func.__name__}
                            )
            return wrapper
        return decorator

    @staticmethod
    def _is_sensitive(key: str) -> bool:
        """Check if a parameter key might contain sensitive information."""
        sensitive_keys = ["password", "token", "secret", "key", "auth"]
        return any(s in key.lower() for s in sensitive_keys)


# Singleton instance
telemetry = Telemetry()
