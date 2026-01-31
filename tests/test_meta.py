import unittest
from unittest.mock import patch, MagicMock
import os
import sys

# Mock dependencies to avoid import errors
sys.modules['z3'] = MagicMock()
sys.modules['torch'] = MagicMock()
sys.modules['torch.nn'] = MagicMock()
sys.modules['torch.distributed'] = MagicMock()
sys.modules['redis'] = MagicMock()

# Import the module under test
# We assume the test runner adds the repo root to sys.path
from src.meta import AutomatedEvoComp

class TestAutomatedEvoComp(unittest.TestCase):

    @patch('src.meta.DistributedEvoContext')
    def test_init_defaults(self, mock_context):
        """Test initialization with default values."""
        # Ensure env vars don't interfere
        with patch.dict(os.environ, {}, clear=True):
            aec = AutomatedEvoComp()

            mock_context.assert_called_once_with(
                world_size=1,
                rank=0,
                redis_host='localhost'
            )

    @patch('src.meta.DistributedEvoContext')
    def test_init_args(self, mock_context):
        """Test initialization with explicit arguments."""
        aec = AutomatedEvoComp(world_size=10, rank=3, redis_host='custom-redis')

        mock_context.assert_called_once_with(
            world_size=10,
            rank=3,
            redis_host='custom-redis'
        )

    @patch('src.meta.DistributedEvoContext')
    def test_init_env_vars(self, mock_context):
        """Test initialization from environment variables."""
        env_vars = {
            'WORLD_SIZE': '5',
            'RANK': '2',
            'REDIS_HOST': 'env-host'
        }
        with patch.dict(os.environ, env_vars):
            aec = AutomatedEvoComp()

            mock_context.assert_called_once_with(
                world_size=5,
                rank=2,
                redis_host='env-host'
            )

    @patch('src.meta.DistributedEvoContext')
    def test_init_precedence(self, mock_context):
        """Test that arguments take precedence over environment variables."""
        env_vars = {
            'WORLD_SIZE': '5',
            'RANK': '2',
            'REDIS_HOST': 'env-host'
        }
        with patch.dict(os.environ, env_vars):
            # Pass explicit args
            aec = AutomatedEvoComp(world_size=8, rank=1, redis_host='arg-host')

            mock_context.assert_called_once_with(
                world_size=8,
                rank=1,
                redis_host='arg-host'
            )

if __name__ == '__main__':
    unittest.main()
