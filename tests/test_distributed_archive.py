import unittest
from unittest.mock import MagicMock
from src.distributed_context import DistributedArchive

class TestDistributedArchive(unittest.TestCase):
    def setUp(self):
        self.mock_redis = MagicMock()
        self.archive = DistributedArchive(self.mock_redis, node_id=0)

    def test_fitnesses_empty(self):
        # Setup mock to return empty list
        self.mock_redis.lrange.return_value = []

        # Check fitnesses
        self.assertEqual(self.archive.fitnesses, [])
        self.mock_redis.lrange.assert_called_with("evocomp:shared_archive:fitnesses", 0, -1)

    def test_add_fitness(self):
        # Add a fitness value
        self.archive.add_fitness(0.95)

        # Check if redis command was called
        self.mock_redis.rpush.assert_called_with("evocomp:shared_archive:fitnesses", 0.95)

    def test_fitnesses_retrieval(self):
        # Setup mock to return some values (as bytes, like Redis does)
        self.mock_redis.lrange.return_value = [b'0.1', b'0.5', b'0.9']

        # Check fitnesses
        fitnesses = self.archive.fitnesses
        self.assertEqual(fitnesses, [0.1, 0.5, 0.9])
        self.assertTrue(all(isinstance(x, float) for x in fitnesses))

if __name__ == '__main__':
    unittest.main()
