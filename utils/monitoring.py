# monitoring_utils.py

from prometheus_client import start_http_server, Gauge
import psutil
import time
import threading

import pynvml

try:
    pynvml.nvmlInit()
    GPU_AVAILABLE = True
except pynvml.NVMLError:
    print("GPU not available. Proceeding without GPU monitoring.")
    GPU_AVAILABLE = False
class TrainingMonitor:
    def __init__(self, port=8000):
        # Training Progress Metrics
        self.epoch_gauge = Gauge('training_epoch', 'Current training epoch')
        self.batch_gauge = Gauge('training_batch', 'Current training batch')
        self.loss_gauge = Gauge('training_loss', 'Current training loss')
        self.val_accuracy_gauge = Gauge('validation_accuracy', 'Current validation accuracy')

        # System Resource Metrics
        self.cpu_usage_gauge = Gauge('cpu_usage_percent', 'CPU Usage Percentage')
        self.memory_usage_gauge = Gauge('memory_usage_percent', 'Memory Usage Percentage')

        # GPU Metrics (if available)
        if GPU_AVAILABLE:
            self.gpu_usage_gauge = Gauge('gpu_usage_percent', 'GPU Usage Percentage')
            self.gpu_memory_gauge = Gauge('gpu_memory_usage_percent', 'GPU Memory Usage Percentage')

        # Start Prometheus server
        start_http_server(port)

        # Start resource monitoring thread
        thread = threading.Thread(target=self._monitor_resources)
        thread.daemon = True
        thread.start()

    def _monitor_resources(self):
        while True:
            self.cpu_usage_gauge.set(psutil.cpu_percent())
            self.memory_usage_gauge.set(psutil.virtual_memory().percent)

            if GPU_AVAILABLE:
                handle = pynvml.nvmlDeviceGetHandleByIndex(0)
                gpu_util = pynvml.nvmlDeviceGetUtilizationRates(handle)
                gpu_mem = pynvml.nvmlDeviceGetMemoryInfo(handle)
                total_mem = gpu_mem.total
                used_mem = gpu_mem.used

                self.gpu_usage_gauge.set(gpu_util.gpu)
                self.gpu_memory_gauge.set((used_mem / total_mem) * 100)

            time.sleep(5)  # Update every 5 seconds

    # Training-specific metric updates
    def update_epoch(self, epoch):
        self.epoch_gauge.set(epoch)

    def update_batch(self, batch):
        self.batch_gauge.set(batch)

    def update_loss(self, loss):
        self.loss_gauge.set(loss)

    def update_val_accuracy(self, accuracy):
        self.val_accuracy_gauge.set(accuracy)
