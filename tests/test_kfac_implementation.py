import unittest
import torch
import torch.nn as nn
from src.evolution import HighDimEvolution

class TestKFAC(unittest.TestCase):
    def setUp(self):
        self.evo = HighDimEvolution()
        self.model = nn.Sequential(
            nn.Linear(10, 5),
            nn.ReLU(),
            nn.Linear(5, 1)
        )
        self.population = [self.model]

    def test_hooks(self):
        self.evo.enable_kfac(self.population)
        self.assertTrue(len(self.evo._kfac_hooks) > 0)

        # Run forward/backward
        x = torch.randn(1, 10)
        y = self.model(x)
        loss = y.mean()
        loss.backward()

        # Check if data captured
        # Layers 0 (Linear) and 2 (Linear) should have data
        l0 = self.model[0]
        l2 = self.model[2]

        self.assertTrue(hasattr(l0, '_kfac_a'), "Layer should have activation storage")
        self.assertTrue(hasattr(l0, '_kfac_g'), "Layer should have gradient storage")
        self.assertTrue(len(l0._kfac_a) > 0, "Activation storage should not be empty")

        self.evo.disable_kfac()
        self.assertEqual(len(self.evo._kfac_hooks), 0)

    def test_block_computation(self):
        self.evo.enable_kfac(self.population)
        x = torch.randn(5, 10) # Batch of 5
        y = self.model(x)
        loss = y.mean()
        loss.backward()

        blocks = self.evo._compute_kfac_blocks(self.population)

        self.assertTrue(len(blocks) > 0)

        # Check dimensions
        # Usually keys are strings like "0", "2" for Sequential

        # Layer 0: Linear(10, 5). Input 10. Bias=True -> Input 11.
        # A matrix: (11, 11).
        # Output 5. G matrix: (5, 5).

        if "0" in blocks:
            A, G = blocks["0"]
            self.assertEqual(A.shape, (11, 11))
            self.assertEqual(G.shape, (5, 5))

        # Layer 2: Linear(5, 1). Input 5. Bias=True -> Input 6.
        # A matrix: (6, 6).
        # Output 1. G matrix: (1, 1).
        if "2" in blocks:
            A, G = blocks["2"]
            self.assertEqual(A.shape, (6, 6))
            self.assertEqual(G.shape, (1, 1))

if __name__ == '__main__':
    unittest.main()
