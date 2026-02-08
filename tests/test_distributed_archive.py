import unittest
from unittest.mock import Mock, call
import redis
from src.distributed_context import DistributedArchive

class TestDistributedArchive(unittest.TestCase):
    def setUp(self):
        self.mock_redis = Mock(spec=redis.Redis)
        self.archive = DistributedArchive(redis_client=self.mock_redis, node_id=0)

    def test_fitnesses_empty(self):
        self.mock_redis.lrange.return_value = []
        self.assertEqual(self.archive.fitnesses, [])
        self.mock_redis.lrange.assert_called_with(self.archive.redis_key, 0, -1)

    def test_add_and_fetch_fitness(self):
        # Initial empty state
        self.mock_redis.lrange.return_value = []
        self.assertEqual(self.archive.fitnesses, [])

        # Add fitness
        self.archive.add_fitness(1.5)
        self.mock_redis.rpush.assert_called_with(self.archive.redis_key, 1.5)

        # Simulate fetch returning one item
        self.mock_redis.lrange.return_value = [b'1.5']
        self.assertEqual(self.archive.fitnesses, [1.5])

    def test_fitnesses_caching(self):
        # 1. First call: fetches initial items
        self.mock_redis.lrange.return_value = [b'1.0', b'2.0']
        fitnesses1 = self.archive.fitnesses
        self.assertEqual(fitnesses1, [1.0, 2.0])
        self.mock_redis.lrange.assert_called_with(self.archive.redis_key, 0, -1)

        # 2. Second call: fetch new items (simulated: one new item)
        # We need to simulate the implementation of caching logic here to test it effectively.
        # But wait, I haven't implemented caching yet. This test is expected to fail or behave differently
        # based on current implementation vs optimized implementation.
        # For the CURRENT implementation, it calls lrange(0, -1) again.

        # Let's write the test assuming the OPTIMIZED behavior will be implemented.
        # If I run this now, it will fail because current implementation always calls lrange(0, -1).

        # Mocking lrange to return only new items if start > 0
        def lrange_side_effect(key, start, end):
            if start == 0:
                return [b'1.0', b'2.0']
            elif start == 2:
                return [b'3.0']
            return []

        self.mock_redis.lrange.side_effect = lrange_side_effect

        # Reset archive for side_effect to take over from start
        self.archive = DistributedArchive(redis_client=self.mock_redis, node_id=0)

        # Call 1
        f1 = self.archive.fitnesses
        self.assertEqual(f1, [1.0, 2.0])
        self.mock_redis.lrange.assert_called_with(self.archive.redis_key, 0, -1)

        # Call 2
        f2 = self.archive.fitnesses
        self.assertEqual(f2, [1.0, 2.0, 3.0])
        # Verify optimized call: start index should be 2 (length of previous list)
        # Note: The current implementation will call (0, -1) and fail this assertion.
        # So this test serves as a TDD test case.
        self.mock_redis.lrange.assert_called_with(self.archive.redis_key, 2, -1)

    def test_redis_error(self):
        self.mock_redis.lrange.side_effect = redis.ConnectionError
        self.assertEqual(self.archive.fitnesses, [])

if __name__ == '__main__':
    unittest.main()
