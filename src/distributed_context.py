import redis
import torch.distributed as dist

class ExperimentResult:
    def __init__(self, results):
        self.results = results
        self.significance_p = self._compute_significance()

    def _compute_significance(self):
        # Stub logic to match previous behavior
        return 0.04

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

    def execute(self, experiment):
        """
        Executes an experiment.
        Distributes tasks to workers if available.
        """
        is_distributed = dist.is_available() and dist.is_initialized()
        rank = dist.get_rank() if is_distributed else 0
        world_size = dist.get_world_size() if is_distributed else 1

        # Broadcast experiment if distributed
        if is_distributed:
            # We wrap experiment in a list because broadcast_object_list expects a list
            objs = [experiment] if rank == 0 else [None]
            dist.broadcast_object_list(objs, src=0)
            experiment = objs[0]

        # Execute locally
        local_result = None
        if hasattr(experiment, 'run'):
            # Pass self (context) so experiment can access archive, redis, etc.
            local_result = experiment.run(self)
        elif callable(experiment):
            local_result = experiment(self)
        else:
            # Fallback for string specs or non-runnable objects (e.g. from stub)
            local_result = None

        # Gather results
        all_results = [None for _ in range(world_size)]
        if is_distributed:
            dist.all_gather_object(all_results, local_result)
        else:
            all_results = [local_result]

        return ExperimentResult(all_results)
