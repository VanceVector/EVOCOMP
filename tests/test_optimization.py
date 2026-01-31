import unittest
import torch
import torch.nn as nn
from src.evolution import select_diverse_pair
from src.manifold import EvoManifoldKernel

class TestOptimization(unittest.TestCase):
    def test_select_diverse_pair(self):
        context = type('Context', (), {'manifold': EvoManifoldKernel(dim=2)})()

        # Archive of 3 models
        m1 = nn.Linear(1, 1)
        m2 = nn.Linear(1, 1)
        m3 = nn.Linear(1, 1)

        # Manually force parameters to ensure they are different
        with torch.no_grad():
            m1.weight.fill_(0.0)
            m1.bias.fill_(0.0)

            m2.weight.fill_(1.0)
            m2.bias.fill_(1.0)

            m3.weight.fill_(2.0)
            m3.bias.fill_(2.0)

        # Expected max distance is between m1 and m3 (0 vs 2)
        pair = select_diverse_pair(context, [m1, m2, m3])
        self.assertEqual(len(pair), 2)

        # Should be m1 and m3
        self.assertTrue((pair[0] is m1 and pair[1] is m3) or (pair[0] is m3 and pair[1] is m1))

    def test_edge_cases(self):
        context = type('Context', (), {'manifold': EvoManifoldKernel(dim=2)})()
        m1 = nn.Linear(1, 1)

        # Size 1
        pair = select_diverse_pair(context, [m1])
        self.assertEqual(pair, (m1, m1))

        # Empty
        pair = select_diverse_pair(context, [])
        self.assertEqual(pair, (None, None))

if __name__ == '__main__':
    unittest.main()
