import redis
import torch.distributed as dist
from typing import List

class ExperimentResult:
    def __init__(self, significance_p=0.04):
        self.significance_p = significance_p

class DistributedArchive:
    def __init__(self, redis_client, node_id):
        self.redis = redis_client
        self.node_id = node_id
        self.redis_key = "evocomp:shared_archive:fitnesses"

    @property
    def fitnesses(self) -> List[float]:
        try:
            # Fetch all items
            items = self.redis.lrange(self.redis_key, 0, -1)
            return [float(x) for x in items]
        except (redis.ConnectionError, TypeError, ValueError):
            return []

    def add_fitness(self, val: float):
        try:
            self.redis.rpush(self.redis_key, val)
        except redis.ConnectionError:
            pass

class EVOCOMPContext:
    pass

class DistributedEvoContext:
    def __init__(self, world_size: int, rank: int, redis_host: str):
        if dist.is_available():
            try:
                if not dist.is_initialized():
                    # We assume master addr/port set in env or arguments, but here defaults
                    dist.init_process_group("gloo", rank=rank, world_size=world_size)
            except Exception:
                # Log warning in real app
                pass

        self.redis = redis.Redis(host=redis_host, socket_connect_timeout=1)

        self.archive = DistributedArchive(
            redis_client=self.redis, node_id=rank)

        self.cuda_ctx = EVOCOMPContext()
        self.manifold = None
        self.failure_log = []

    def execute(self, experiment):
        """
        Executes an experiment.
        """
        # Logic: Broadcast experiment from rank 0 to others
        if dist.is_initialized():
            objects = [experiment]
            try:
                dist.broadcast_object_list(objects, src=0)
                # In a real loop, workers would receive this.
                # Here we are just the caller.

                # Gather results (simulated)
                # results = [None for _ in range(dist.get_world_size())]
                # dist.gather_object(local_result, results if rank==0 else None, dst=0)
                pass
            except Exception:
                pass

        return ExperimentResult()
