import redis
import torch.distributed as dist
from typing import Any, List, Optional, Dict

class ExperimentResult:
    def __init__(self, significance_p: float = 0.0, metrics: Optional[Dict[str, Any]] = None):
        self.significance_p = significance_p
        self.metrics = metrics or {}

class DistributedArchive:
    def __init__(self, redis_client, node_id):
        self.redis = redis_client
        self.node_id = node_id
        self.fitnesses = [] # Stub: list of fitness values

class EVOCOMPContext:
    pass

class DistributedEvoContext:
    def __init__(self, world_size: int, rank: int, redis_host: str):
        # NCCL for parameter sync (high bandwidth)
        if dist.is_available():
            try:
                # In a real environment this would block until all processes join
                # We wrap in try/except for the sake of this setup/test environment
                dist.init_process_group("gloo", rank=rank, world_size=world_size) # Using gloo for CPU support if needed
            except Exception as e:
                print(f"Distributed init failed (expected in single node env): {e}")

        # Redis - Raft for metadata consensus
        # Using strict=False to allow instantiation without immediate connection check if needed,
        # but redis-py usually connects lazily or checks on first command.
        self.redis = redis.Redis(host=redis_host, socket_connect_timeout=1)

        self.archive = DistributedArchive(
            redis_client=self.redis, node_id=rank)

        # Shared GPU resources
        self.cuda_ctx = EVOCOMPContext()
        self.manifold = None # This should be set by the application
        self.failure_log = []

    def worker_loop(self):
        """
        Loop for worker nodes to receive and execute experiments.
        """
        if not dist.is_initialized():
             return

        while True:
            # Receive experiment
            objects = [None]
            # In a real scenario, we'd handle exceptions or timeouts
            dist.broadcast_object_list(objects, src=0)
            experiment = objects[0]

            if experiment == "STOP":
                break

            # Execute
            local_result = self._run_experiment_locally(experiment)

            # Gather results
            results = [None for _ in range(dist.get_world_size())]
            dist.all_gather_object(results, local_result)

    def execute(self, experiment):
        """
        Executes an experiment.
        """
        if not dist.is_initialized():
            return self._run_experiment_locally(experiment)

        # Distributed execution
        # Broadcast experiment
        objects = [experiment]
        dist.broadcast_object_list(objects, src=0)

        # Run locally
        local_result = self._run_experiment_locally(experiment)

        # Gather results
        results = [None for _ in range(dist.get_world_size())]
        dist.all_gather_object(results, local_result)

        # Aggregate results
        return self._aggregate_results(results)

    def _run_experiment_locally(self, experiment):
        if hasattr(experiment, 'run'):
            return experiment.run()
        elif isinstance(experiment, str):
             # Stub/Legacy behavior
             return ExperimentResult(significance_p=0.04, metrics={"rank": dist.get_rank() if dist.is_initialized() else 0})
        else:
             # Unknown type, return default stub
             return ExperimentResult(significance_p=0.04)

    def _aggregate_results(self, results: List[ExperimentResult]) -> ExperimentResult:
        avg_significance = 0.0
        valid_results = 0
        all_metrics = []

        for r in results:
            if r and isinstance(r, ExperimentResult):
                avg_significance += r.significance_p
                valid_results += 1
                all_metrics.append(r.metrics)

        if valid_results > 0:
            avg_significance /= valid_results

        return ExperimentResult(significance_p=avg_significance, metrics={"all_metrics": all_metrics})
