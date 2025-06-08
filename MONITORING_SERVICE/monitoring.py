from prometheus_client import Counter, Histogram, Gauge, make_asgi_app
from prometheus_fastapi_instrumentator import Instrumentator, metrics
import time
import pynvml
from typing import Dict, Any
from prometheus_fastapi_instrumentator.metrics import Info

class GPUMetricsManager:
    """Manages GPU-related metrics"""
    def __init__(self, service_name: str = "default"):
        try:
            pynvml.nvmlInit()
            self.gpu_handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            self.has_gpu = True
        except:
            print("Warning: NVIDIA GPU monitoring not available")
            self.has_gpu = False

        # GPU metrics
        self.gpu_memory_used = Gauge(
            f'{service_name}_gpu_memory_used_bytes',
            'GPU memory used in bytes'
        )
        
        self.gpu_utilization = Gauge(
            f'{service_name}_gpu_utilization_percent',
            'GPU utilization percentage'
        )

    def update_gpu_metrics(self):
        """Update GPU metrics if available"""
        if self.has_gpu:
            try:
                # Get GPU utilization
                utilization = pynvml.nvmlDeviceGetUtilizationRates(self.gpu_handle)
                self.gpu_utilization.set(utilization.gpu)

                # Get memory info
                memory_info = pynvml.nvmlDeviceGetMemoryInfo(self.gpu_handle)
                self.gpu_memory_used.set(memory_info.used)  # In bytes
            except:
                print("Warning: Failed to update GPU metrics")

class ModelMetricsManager:
    """Manages ML model-related metrics"""
    def __init__(self, service_name: str = "default"):
        # Model inference metrics
        self.inference_gpu_time = Histogram(
            f'{service_name}_model_inference_gpu_seconds',
            'Time spent on GPU for model inference',
            buckets=[0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0, 2.0]  # More granular buckets for GPU time
        )

        self.inference_cpu_time = Histogram(
            f'{service_name}_model_inference_cpu_seconds',
            'Time spent on CPU for model inference',
            buckets=[0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0, 2.0]  # More granular buckets for CPU time
        )

        self.model_confidence = Histogram(
            f'{service_name}_model_confidence_score',
            'Confidence scores of model predictions',
            buckets=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99]
        )

    def record_inference_metrics(self, gpu_time: float, cpu_time: float, confidence: float = None):
        """Record metrics for model inference"""
        self.inference_gpu_time.observe(gpu_time)
        self.inference_cpu_time.observe(cpu_time)
        if confidence is not None:
            self.model_confidence.observe(confidence)

class TrainingMetricsManager:
    """Manages training-specific metrics"""
    def __init__(self):
        self.epoch_loss = Gauge(
            'training_epoch_loss',
            'Training loss per epoch'
        )
        self.validation_accuracy = Gauge(
            'training_validation_accuracy',
            'Validation accuracy'
        )
        self.learning_rate = Gauge(
            'training_learning_rate',
            'Current learning rate'
        )

    def record_training_metrics(self, loss: float = None, accuracy: float = None, lr: float = None):
        """Record training metrics"""
        if loss is not None:
            self.epoch_loss.set(loss)
        if accuracy is not None:
            self.validation_accuracy.set(accuracy)
        if lr is not None:
            self.learning_rate.set(lr)

def model_prediction_time():
    """Custom metric for FastAPI instrumentator to track model prediction time"""
    def instrumentation(info: Info):
        if info.request.url.path == "/predict":
            start = time.time()
            yield
            duration = time.time() - start
            # These metrics will be automatically exposed by FastAPI instrumentator
            info.metric(
                name="model_prediction_duration_seconds",
                documentation="Time spent processing model prediction",
                value=duration
            )
        else:
            yield

    return instrumentation

def setup_instrumentator() -> Instrumentator:
    """Configure FastAPI instrumentator with custom metrics"""
    instrumentator = Instrumentator(
        should_group_status_codes=True,
        should_ignore_untemplated=True,
        excluded_handlers=["/metrics"]
    )

    # Add default metrics
    instrumentator.add(
        metrics.latency(
            should_include_handler=True,
            should_include_method=True,
            should_include_status=True,
            metric_namespace="fastapi",
            metric_subsystem="app",
            metric_name="request_duration_seconds",
        )
    ).add(
        metrics.requests(
            should_include_handler=True,
            should_include_method=True,
            should_include_status=True,
            metric_namespace="fastapi",
            metric_subsystem="app",
            metric_name="requests_total",
        )
    )

    return instrumentator

# Create metrics app
metrics_app = make_asgi_app()

# Create instances for API monitoring
gpu_metrics = GPUMetricsManager(service_name="api")
model_metrics = ModelMetricsManager(service_name="api")

# Training metrics will be created in pl_optuna.py 