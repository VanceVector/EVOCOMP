import unittest
from unittest.mock import MagicMock, patch
import sys
import torch

# Mock torch.distributed if not available or just to control it
class MockDist:
    def is_available(self):
        return True
    def is_initialized(self):
        return True
    def get_rank(self):
        return 0
    def get_world_size(self):
        return 2
    def broadcast_object_list(self, obj_list, src=0):
        # Simulate broadcast: in a real test we can't easily sync processes,
        # but for unit testing logic flow, we can just ensure it's called.
        pass
    def all_gather_object(self, obj_list, obj):
        # Simulate gathering: fill obj_list with obj
        for i in range(len(obj_list)):
            obj_list[i] = obj

# We need to patch torch.distributed before importing DistributedEvoContext if we want to mock initialization,
# but here we mainly want to test execute().

from src.distributed_context import DistributedEvoContext

class TestDistributedExecute(unittest.TestCase):
    def setUp(self):
        self.redis_mock = MagicMock()

    @patch('src.distributed_context.dist')
    def test_execute_single_node(self, mock_dist):
        # Setup mock for single node
        mock_dist.is_available.return_value = False
        mock_dist.is_initialized.return_value = False

        ctx = DistributedEvoContext(world_size=1, rank=0, redis_host='localhost')
        ctx.redis = self.redis_mock # Inject mock redis

        # Define a simple experiment
        class SimpleExperiment:
            def run(self, context):
                return "success"

        result = ctx.execute(SimpleExperiment())

        self.assertTrue(hasattr(result, 'significance_p'))
        self.assertEqual(result.results, ["success"])

    @patch('src.distributed_context.dist')
    def test_execute_distributed_rank_0(self, mock_dist):
        # Setup mock for distributed rank 0
        mock_dist.is_available.return_value = True
        mock_dist.is_initialized.return_value = True
        mock_dist.get_rank.return_value = 0
        mock_dist.get_world_size.return_value = 2

        # Mock broadcast and gather
        def side_effect_broadcast(obj_list, src=0):
            # In rank 0, obj_list has the experiment. We leave it as is.
            pass
        mock_dist.broadcast_object_list.side_effect = side_effect_broadcast

        def side_effect_gather(obj_list, obj):
            # Simulate receiving "success" from all nodes
            for i in range(len(obj_list)):
                obj_list[i] = "success"
        mock_dist.all_gather_object.side_effect = side_effect_gather

        ctx = DistributedEvoContext(world_size=2, rank=0, redis_host='localhost')
        ctx.redis = self.redis_mock

        class SimpleExperiment:
            def run(self, context):
                return "success"

        result = ctx.execute(SimpleExperiment())

        self.assertTrue(hasattr(result, 'significance_p'))
        self.assertEqual(result.results, ["success", "success"])

        mock_dist.broadcast_object_list.assert_called()
        mock_dist.all_gather_object.assert_called()

    @patch('src.distributed_context.dist')
    def test_execute_callable(self, mock_dist):
        mock_dist.is_available.return_value = False

        ctx = DistributedEvoContext(world_size=1, rank=0, redis_host='localhost')
        ctx.redis = self.redis_mock

        def simple_func(context):
            return "callable_success"

        result = ctx.execute(simple_func)
        self.assertEqual(result.results, ["callable_success"])

    @patch('src.distributed_context.dist')
    def test_execute_stub_string(self, mock_dist):
        mock_dist.is_available.return_value = False

        ctx = DistributedEvoContext(world_size=1, rank=0, redis_host='localhost')
        ctx.redis = self.redis_mock

        # Test with the string stub from meta.py
        result = ctx.execute("Experiment Spec")
        # Should result in None result but valid object
        self.assertTrue(hasattr(result, 'significance_p'))
        self.assertEqual(result.results, [None])

if __name__ == '__main__':
    unittest.main()
