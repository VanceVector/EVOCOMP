"""
Health monitoring system for EVOCOMP distributed components.
"""

import time
import psutil
import threading
import json
import socket
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Callable
from enum import Enum
from datetime import datetime
import logging

logger = logging.getLogger(__name__)

class HealthStatus(Enum):
    """Health status enumeration."""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    UNKNOWN = "unknown"

@dataclass
class HealthMetric:
    """Individual health metric."""
    name: str
    value: float
    unit: str
    threshold_warning: float
    threshold_critical: float
    timestamp: datetime

    def status(self) -> HealthStatus:
        if self.value > self.threshold_critical:
            return HealthStatus.UNHEALTHY
        elif self.value > self.threshold_warning:
            return HealthStatus.DEGRADED
        return HealthStatus.HEALTHY

@dataclass
class ComponentHealth:
    """Health status of a single component."""
    component_name: str
    status: HealthStatus
    metrics: Dict[str, HealthMetric]
    last_updated: datetime
    dependencies: List[str]

    def to_dict(self) -> dict:
        return {
            "component": self.component_name,
            "status": self.status.value,
            "metrics": {k: asdict(v) for k, v in self.metrics.items()},
            "last_updated": self.last_updated.isoformat(),
            "dependencies": self.dependencies,
        }

class HealthChecker:
    """
    Main health checking system with support for distributed monitoring.

    Features:
    - Resource monitoring (CPU, memory, GPU, disk, network)
    - Service dependency checking
    - Custom health check registration
    - Automatic remediation suggestions
    - Historical health data tracking
    """

    def __init__(self, node_id: int, cluster_size: int):
        self.node_id = node_id
        self.cluster_size = cluster_size
        self.components: Dict[str, ComponentHealth] = {}
        self.custom_checks: Dict[str, Callable[[], bool]] = {}
        self.history: List[Dict] = []
        self.lock = threading.RLock()

        # Register default system checks
        self._register_default_checks()

        # Start background monitoring thread
        self.monitor_thread = threading.Thread(
            target=self._monitoring_loop,
            daemon=True,
            name=f"health-monitor-{node_id}"
        )
        self.monitor_thread.start()

    def _register_default_checks(self):
        """Register default system health checks."""
        # System resources
        self.register_component("system", ["cpu", "memory", "disk", "network"])

        # EVOCOMP components
        self.register_component("evolution_engine", ["system", "redis"])
        self.register_component("manifold_system", ["system"])
        self.register_component("verification", ["system", "z3"])
        self.register_component("distributed", ["system", "network", "nccl"])

        # Services
        self.register_component("redis", ["system", "network"])
        self.register_component("nccl", ["system", "network", "cuda"])
        self.register_component("cuda", ["system"])
        self.register_component("z3", ["system"])

    def register_component(self, name: str, dependencies: List[str] = None):
        """Register a new component for health monitoring."""
        with self.lock:
            self.components[name] = ComponentHealth(
                component_name=name,
                status=HealthStatus.UNKNOWN,
                metrics={},
                last_updated=datetime.now(),
                dependencies=dependencies or []
            )

    def register_custom_check(self, name: str, check_func: Callable[[], bool]):
        """Register a custom health check function."""
        self.custom_checks[name] = check_func

    def _check_system_metrics(self) -> Dict[str, HealthMetric]:
        """Collect system-level health metrics."""
        metrics = {}

        # CPU usage
        cpu_percent = psutil.cpu_percent(interval=0.1)
        metrics["cpu"] = HealthMetric(
            name="cpu_usage",
            value=cpu_percent,
            unit="percent",
            threshold_warning=80.0,
            threshold_critical=95.0,
            timestamp=datetime.now()
        )

        # Memory usage
        memory = psutil.virtual_memory()
        metrics["memory"] = HealthMetric(
            name="memory_usage",
            value=memory.percent,
            unit="percent",
            threshold_warning=85.0,
            threshold_critical=95.0,
            timestamp=datetime.now()
        )

        # Disk usage (root filesystem)
        try:
            disk = psutil.disk_usage('/')
            metrics["disk"] = HealthMetric(
                name="disk_usage",
                value=disk.percent,
                unit="percent",
                threshold_warning=90.0,
                threshold_critical=98.0,
                timestamp=datetime.now()
            )
        except Exception as e:
            logger.warning(f"Failed to check disk usage: {e}")

        # Network connectivity
        try:
            # Check basic connectivity
            socket.create_connection(("8.8.8.8", 53), timeout=2).close()
            network_latency = 1.0  # Good
        except (socket.timeout, ConnectionError):
            network_latency = 1000.0  # Bad

        metrics["network"] = HealthMetric(
            name="network_latency",
            value=network_latency,
            unit="ms",
            threshold_warning=100.0,
            threshold_critical=500.0,
            timestamp=datetime.now()
        )

        # GPU metrics if available
        try:
            import pynvml
            pynvml.nvmlInit()
            try:
                gpu_count = pynvml.nvmlDeviceGetCount()

                for i in range(gpu_count):
                    handle = pynvml.nvmlDeviceGetHandleByIndex(i)
                    util = pynvml.nvmlDeviceGetUtilizationRates(handle)
                    memory_info = pynvml.nvmlDeviceGetMemoryInfo(handle)

                    metrics[f"gpu_{i}_util"] = HealthMetric(
                        name=f"gpu_{i}_utilization",
                        value=util.gpu,
                        unit="percent",
                        threshold_warning=90.0,
                        threshold_critical=98.0,
                        timestamp=datetime.now()
                    )

                    memory_percent = (memory_info.used / memory_info.total) * 100
                    metrics[f"gpu_{i}_memory"] = HealthMetric(
                        name=f"gpu_{i}_memory",
                        value=memory_percent,
                        unit="percent",
                        threshold_warning=90.0,
                        threshold_critical=98.0,
                        timestamp=datetime.now()
                    )
            finally:
                pynvml.nvmlShutdown()
        except ImportError:
            logger.debug("pynvml not available, skipping GPU metrics")
        except Exception as e:
            logger.warning(f"Failed to check GPU metrics: {e}")

        return metrics

    def _check_service_dependencies(self, component: ComponentHealth) -> bool:
        """Check if all dependencies are healthy."""
        with self.lock:
            for dep_name in component.dependencies:
                if dep_name in self.components:
                    dep_status = self.components[dep_name].status
                    if dep_status in [HealthStatus.UNHEALTHY, HealthStatus.UNKNOWN]:
                        logger.warning(f"Component {component.component_name} has unhealthy dependency: {dep_name}")
                        return False
        return True

    def _update_component_health(self, component_name: str):
        """Update health status for a specific component."""
        with self.lock:
            if component_name not in self.components:
                return

            component = self.components[component_name]

            # Get metrics based on component type
            if component_name == "system":
                component.metrics = self._check_system_metrics()
            elif component_name in self.custom_checks:
                # Run custom check
                try:
                    is_healthy = self.custom_checks[component_name]()
                    component.metrics["custom"] = HealthMetric(
                        name="custom_check",
                        value=0.0 if is_healthy else 1.0,
                        unit="boolean",
                        threshold_warning=0.5,
                        threshold_critical=0.9,
                        timestamp=datetime.now()
                    )
                except Exception as e:
                    logger.error(f"Custom check failed for {component_name}: {e}")
                    component.metrics["custom"] = HealthMetric(
                        name="custom_check",
                        value=1.0,
                        unit="boolean",
                        threshold_warning=0.5,
                        threshold_critical=0.9,
                        timestamp=datetime.now()
                    )

            # Determine overall status
            # Check dependencies first
            dependencies_healthy = self._check_service_dependencies(component)

            if not dependencies_healthy:
                component.status = HealthStatus.UNHEALTHY
            elif not component.metrics:
                # If no metrics but dependencies are healthy, assume healthy
                component.status = HealthStatus.HEALTHY
            else:
                # Check all metrics
                worst_status = HealthStatus.HEALTHY
                for metric in component.metrics.values():
                    metric_status = metric.status()
                    if metric_status == HealthStatus.UNHEALTHY:
                        worst_status = HealthStatus.UNHEALTHY
                        break
                    elif metric_status == HealthStatus.DEGRADED and worst_status == HealthStatus.HEALTHY:
                        worst_status = HealthStatus.DEGRADED

                component.status = worst_status

            component.last_updated = datetime.now()

    def _monitoring_loop(self):
        """Background thread that periodically updates health status."""
        logger.info("Starting health monitoring loop")

        while True:
            try:
                # Update all components
                with self.lock:
                    component_names = list(self.components.keys())

                for name in component_names:
                    self._update_component_health(name)

                # Store history (keep last 1000 entries)
                snapshot = self.get_cluster_health()
                self.history.append(snapshot)
                if len(self.history) > 1000:
                    self.history.pop(0)

                # Log if anything is unhealthy
                for component in self.components.values():
                    if component.status == HealthStatus.UNHEALTHY:
                        logger.error(f"Component {component.component_name} is unhealthy")

                time.sleep(5)  # Check every 5 seconds

            except Exception as e:
                logger.error(f"Health monitoring loop error: {e}")
                time.sleep(10)

    def get_component_health(self, component_name: str) -> Optional[ComponentHealth]:
        """Get health status for a specific component."""
        with self.lock:
            return self.components.get(component_name)

    def get_cluster_health(self) -> Dict:
        """Get aggregated health status for the entire cluster."""
        with self.lock:
            components = {
                name: comp.to_dict()
                for name, comp in self.components.items()
            }

            # Calculate overall status
            status_counts = {
                HealthStatus.HEALTHY.value: 0,
                HealthStatus.DEGRADED.value: 0,
                HealthStatus.UNHEALTHY.value: 0,
                HealthStatus.UNKNOWN.value: 0
            }

            for comp in self.components.values():
                status_counts[comp.status.value] += 1

            return {
                "node_id": self.node_id,
                "timestamp": datetime.now().isoformat(),
                "components": components,
                "status_counts": status_counts,
                "total_components": len(self.components),
            }

    def get_remediation_suggestions(self) -> List[Dict]:
        """Get suggestions for fixing unhealthy components."""
        suggestions = []

        with self.lock:
            for component in self.components.values():
                if component.status == HealthStatus.UNHEALTHY:
                    for metric_name, metric in component.metrics.items():
                        if metric.status() == HealthStatus.UNHEALTHY:
                            suggestion = self._generate_suggestion(
                                component.component_name,
                                metric_name,
                                metric
                            )
                            if suggestion:
                                suggestions.append(suggestion)

        return suggestions

    def _generate_suggestion(self, component: str, metric: str, health_metric: HealthMetric) -> Dict:
        """Generate remediation suggestion based on metric."""
        base_suggestion = {
            "component": component,
            "metric": metric,
            "current_value": health_metric.value,
            "threshold": health_metric.threshold_critical,
            "timestamp": health_metric.timestamp.isoformat()
        }

        if metric == "cpu_usage":
            return {
                **base_suggestion,
                "suggestion": "Reduce CPU load by limiting concurrent processes or optimizing code",
                "action": "Check for runaway processes, reduce batch sizes, or add more nodes"
            }
        elif metric == "memory_usage":
            return {
                **base_suggestion,
                "suggestion": "Free up memory by clearing caches or reducing model sizes",
                "action": "Clear PyTorch cache, reduce population size, or add swap space"
            }
        elif metric == "disk_usage":
            return {
                **base_suggestion,
                "suggestion": "Free up disk space by cleaning old checkpoints and logs",
                "action": f"Run cleanup script: python scripts/cleanup.py --keep-last 5"
            }
        elif metric == "network_latency":
            return {
                **base_suggestion,
                "suggestion": "Network connectivity issues detected",
                "action": "Check network cables, switch configuration, or reduce network traffic"
            }
        elif "gpu" in metric:
            return {
                **base_suggestion,
                "suggestion": "GPU resource constraint detected",
                "action": "Reduce batch size, enable mixed precision, or spread load across more GPUs"
            }

        return base_suggestion

    def export_health_data(self, filepath: str):
        """Export health history to JSON file."""
        with open(filepath, 'w') as f:
            json.dump({
                "node_id": self.node_id,
                "history": self.history,
                "exported_at": datetime.now().isoformat()
            }, f, indent=2)

    def wait_for_healthy(self, component_name: str, timeout: int = 300) -> bool:
        """Wait for a component to become healthy (blocking)."""
        start_time = time.time()

        while time.time() - start_time < timeout:
            health = self.get_component_health(component_name)
            if health and health.status == HealthStatus.HEALTHY:
                return True
            time.sleep(1)

        return False
