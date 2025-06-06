from prometheus_client import Counter, Histogram, Gauge, make_asgi_app
import time
import pynvml
from typing import Dict, Any

class MetricsManager:
    def __init__(self, service_name: str = "default"):
        # Initialize NVML for GPU metrics
        try:
            pynvml.nvmlInit()
            self.gpu_handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            self.has_gpu = True
        except:
            print("Warning: NVIDIA GPU monitoring not available")
            self.has_gpu = False

        # Request metrics
        self.request_counter = Counter(
            f'{service_name}_http_requests_total',
            'Total number of HTTP requests',
            ['method', 'endpoint', 'status']
        )
        
        self.request_latency = Histogram(
            f'{service_name}_http_request_duration_seconds',
            'HTTP request latency in seconds',
            ['method', 'endpoint'],
            buckets=[0.01, 0.05, 0.1, 0.5, 1.0, 2.0, 5.0]
        )

        # Model metrics
        self.inference_gpu_time = Histogram(
            f'{service_name}_model_inference_gpu_seconds',
            'Time spent on GPU for model inference/training',
            buckets=[0.01, 0.05, 0.1, 0.5, 1.0, 2.0]
        )

        self.inference_cpu_time = Histogram(
            f'{service_name}_model_inference_cpu_seconds',
            'Time spent on CPU for model inference/training',
            buckets=[0.01, 0.05, 0.1, 0.5, 1.0, 2.0]
        )

        self.model_confidence = Histogram(
            f'{service_name}_model_confidence_score',
            'Confidence scores of model predictions',
            buckets=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99]
        )

        # Training specific metrics
        if service_name == "training":
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

        # GPU metrics
        self.gpu_memory_used = Gauge(
            f'{service_name}_gpu_memory_used_bytes',
            'GPU memory used in bytes'
        )
        
        self.gpu_utilization = Gauge(
            f'{service_name}_gpu_utilization_percent',
            'GPU utilization percentage'
        )

    def record_request_metrics(self, method: str, endpoint: str, status: int, duration: float):
        """Record metrics for an HTTP request"""
        self.request_counter.labels(method=method, endpoint=endpoint, status=status).inc()
        self.request_latency.labels(method=method, endpoint=endpoint).observe(duration)

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

    def record_inference_metrics(self, gpu_time: float, cpu_time: float, confidence: float = None):
        """Record metrics for model inference/training"""
        self.inference_gpu_time.observe(gpu_time)
        self.inference_cpu_time.observe(cpu_time)
        if confidence is not None:
            self.model_confidence.observe(confidence)

    def record_training_metrics(self, loss: float = None, accuracy: float = None, lr: float = None):
        """Record training specific metrics"""
        if hasattr(self, 'epoch_loss') and loss is not None:
            self.epoch_loss.set(loss)
        if hasattr(self, 'validation_accuracy') and accuracy is not None:
            self.validation_accuracy.set(accuracy)
        if hasattr(self, 'learning_rate') and lr is not None:
            self.learning_rate.set(lr)

# Global metrics manager instances for different services
api_metrics_manager = MetricsManager(service_name="api")
training_metrics_manager = MetricsManager(service_name="training") 