import unittest
import torch
import numpy as np
from src.evolution import HighDimEvolution

class TestEvolutionAdvantages(unittest.TestCase):
    def setUp(self):
        self.evo = HighDimEvolution()

    def test_standard_case(self):
        fitnesses = [1.0, 2.0, 3.0, 4.0, 5.0]
        # Mean = 3.0, Std (unbiased) = 1.5811
        # Expected: [-1.2649, -0.6324, 0.0, 0.6324, 1.2649]
        advantages = self.evo._compute_advantages(fitnesses)

        self.assertIsInstance(advantages, torch.Tensor)
        self.assertEqual(advantages.shape, (5,))
        self.assertAlmostEqual(advantages.mean().item(), 0.0, places=5)
        self.assertAlmostEqual(advantages.std().item(), 1.0, places=5)

    def test_constant_fitness(self):
        fitnesses = [10.0, 10.0, 10.0]
        advantages = self.evo._compute_advantages(fitnesses)
        # Should be zeros
        self.assertTrue(torch.all(advantages == 0))

    def test_single_fitness(self):
        fitnesses = [5.0]
        advantages = self.evo._compute_advantages(fitnesses)
        # Should be zero
        self.assertTrue(torch.all(advantages == 0))

    def test_empty_fitness(self):
        fitnesses = []
        advantages = self.evo._compute_advantages(fitnesses)
        self.assertEqual(len(advantages), 0)

    def test_negative_fitness(self):
        fitnesses = [-5.0, -1.0, -3.0]
        advantages = self.evo._compute_advantages(fitnesses)
        self.assertAlmostEqual(advantages.mean().item(), 0.0, places=5)
        self.assertAlmostEqual(advantages.std().item(), 1.0, places=5)

if __name__ == '__main__':
    unittest.main()
