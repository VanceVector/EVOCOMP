
import unittest
import torch
import torch.nn as nn
from src.evolution import select_diverse_pair

class MockManifold:
    def __init__(self, embeddings_map):
        self.embeddings_map = embeddings_map

    def embed(self, model):
        return self.embeddings_map[model]

    def geodesic_distance(self, emb_i, emb_j):
        # Euclidean distance implementation for the test
        # Ensure we can handle both tensor and list inputs if the implementation changes
        if isinstance(emb_i, list):
            emb_i = torch.tensor(emb_i)
        if isinstance(emb_j, list):
            emb_j = torch.tensor(emb_j)

        return torch.dist(emb_i, emb_j).item()

class MockContext:
    def __init__(self, embeddings_map):
        self.manifold = MockManifold(embeddings_map)

class TestOptimization(unittest.TestCase):
    def test_select_diverse_pair_correctness(self):
        # Create dummy models
        m1 = nn.Linear(1, 1)
        m2 = nn.Linear(1, 1)
        m3 = nn.Linear(1, 1)

        archive = [m1, m2, m3]

        # Define embeddings: points on a line
        # m1: [0.0]
        # m2: [1.0]
        # m3: [10.0]
        embeddings_map = {
            m1: [0.0],
            m2: [1.0],
            m3: [10.0]
        }

        ctx = MockContext(embeddings_map)

        # Expected max distance is between m1 and m3 (distance 10.0)
        pair = select_diverse_pair(ctx, archive)

        # Check that we got m1 and m3 (order doesn't matter for the pair set,
        # but the function returns a tuple. Usually (min_idx, max_idx) based on loop order
        # or just the pair.

        self.assertTrue((pair[0] == m1 and pair[1] == m3) or (pair[0] == m3 and pair[1] == m1),
                        f"Expected (m1, m3), got {pair}")

    def test_select_diverse_pair_tie(self):
        # Test tie breaking behavior if needed, or just that it returns a valid pair
        m1 = nn.Linear(1, 1)
        m2 = nn.Linear(1, 1)
        m3 = nn.Linear(1, 1)

        archive = [m1, m2, m3]

        # Equilateral triangle
        # m1: [0, 0]
        # m2: [1, 0]
        # m3: [0.5, sqrt(3)/2] ~ [0.5, 0.866]
        # All distances are 1.0.

        embeddings_map = {
            m1: [0.0, 0.0],
            m2: [1.0, 0.0],
            m3: [0.5, 0.86602540378]
        }

        ctx = MockContext(embeddings_map)
        pair = select_diverse_pair(ctx, archive)

        self.assertIsNotNone(pair)
        self.assertEqual(len(pair), 2)
        self.assertIn(pair[0], archive)
        self.assertIn(pair[1], archive)
        self.assertNotEqual(pair[0], pair[1])

    def test_select_diverse_pair_tensors(self):
        # Test when embed returns tensors
        m1 = nn.Linear(1, 1)
        m2 = nn.Linear(1, 1)

        archive = [m1, m2]

        embeddings_map = {
            m1: torch.tensor([0.0]),
            m2: torch.tensor([1.0])
        }

        ctx = MockContext(embeddings_map)
        pair = select_diverse_pair(ctx, archive)

        self.assertEqual(pair, (m1, m2))

if __name__ == '__main__':
    unittest.main()
