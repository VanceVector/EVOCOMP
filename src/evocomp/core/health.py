import threading
import time
import psutil
import pynvml
from typing import Dict, Any

class HealthChecker:
    def __init__(self, interval: float = 1.0):
        self.interval = interval
        self.running = False
        self.thread = None
        self.metrics: Dict[str, float] = {}
        self.lock = threading.Lock()

    def start(self):
        if self.running:
            return
        self.running = True
        self.thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self.thread.start()

    def stop(self):
        self.running = False
        if self.thread:
            self.thread.join()

    def _monitor_loop(self):
        while self.running:
            new_metrics = {}

            # System Metrics (Higher is worse, normalized 0-1)
            # CPU
            new_metrics['cpu'] = psutil.cpu_percent(interval=None) / 100.0

            # Memory
            mem = psutil.virtual_memory()
            new_metrics['memory'] = mem.percent / 100.0

            # Disk
            disk = psutil.disk_usage('/')
            new_metrics['disk'] = disk.percent / 100.0

            # GPU
            try:
                pynvml.nvmlInit()
                device_count = pynvml.nvmlDeviceGetCount()
                if device_count > 0:
                    handle = pynvml.nvmlDeviceGetHandleByIndex(0)
                    util = pynvml.nvmlDeviceGetUtilizationRates(handle)
                    # util.gpu is 0-100
                    new_metrics['gpu'] = util.gpu / 100.0
                else:
                    new_metrics['gpu'] = 0.0 # No GPU = No load
                pynvml.nvmlShutdown()
            except Exception:
                # If NVML fails, assume healthy or not present
                new_metrics['gpu'] = 0.0
                try:
                    pynvml.nvmlShutdown()
                except:
                    pass

            with self.lock:
                self.metrics = new_metrics

            time.sleep(self.interval)

    def check_health(self) -> Dict[str, float]:
        with self.lock:
            return self.metrics.copy()

    def is_healthy(self, threshold: float = 0.9) -> bool:
        with self.lock:
            for k, v in self.metrics.items():
                if v > threshold:
                    return False
        return True
