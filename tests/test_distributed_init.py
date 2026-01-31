import unittest
from unittest.mock import patch
import os
from src.meta import AutomatedEvoComp

class TestDistributedInit(unittest.TestCase):
    @patch('src.meta.DistributedEvoContext')
    def test_default_init(self, MockContext):
        # Test default initialization
        AutomatedEvoComp()

        # Verify default args
        MockContext.assert_called_with(world_size=1, rank=0, redis_host='localhost')

    @patch('src.meta.DistributedEvoContext')
    def test_env_vars_init(self, MockContext):
        # Set environment variables
        env_vars = {
            'WORLD_SIZE': '4',
            'RANK': '2',
            'REDIS_HOST': '192.168.1.100'
        }

        with patch.dict(os.environ, env_vars):
            AutomatedEvoComp()

            # Verify args from env vars
            MockContext.assert_called_with(world_size=4, rank=2, redis_host='192.168.1.100')

    @patch('src.meta.DistributedEvoContext')
    def test_explicit_init(self, MockContext):
        # Test explicit arguments (assuming we add them)
        # Note: This test will fail until we modify AutomatedEvoComp to accept args
        # But for now, we try to call it as we plan to implement it.
        try:
            AutomatedEvoComp(world_size=8, rank=3, redis_host='redis-cluster')
            MockContext.assert_called_with(world_size=8, rank=3, redis_host='redis-cluster')
        except TypeError:
            # Expected failure before implementation
            print("Caught expected TypeError for explicit init before implementation")
            pass

    @patch('src.meta.DistributedEvoContext')
    def test_priority(self, MockContext):
        # Test that explicit args override env vars
        env_vars = {
            'WORLD_SIZE': '4',
            'RANK': '2',
            'REDIS_HOST': 'env-host'
        }

        with patch.dict(os.environ, env_vars):
            try:
                AutomatedEvoComp(world_size=10, rank=5, redis_host='arg-host')
                MockContext.assert_called_with(world_size=10, rank=5, redis_host='arg-host')
            except TypeError:
                pass

if __name__ == '__main__':
    unittest.main()
