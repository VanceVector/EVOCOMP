import torch
import torch.nn as nn
import math
import unittest
from src.evolution import HighDimEvolution

class TestHighDimEvolutionTRPO(unittest.TestCase):
    def test_trpo_step_with_tensors(self):
        # Setup
        evo = HighDimEvolution()

        # Mock methods
        torch.manual_seed(42)
        d_in = 5
        d_out = 3
        W = torch.randn(d_out, d_in)
        A_root = torch.randn(d_in, d_in)
        A = A_root @ A_root.T
        G_root = torch.randn(d_out, d_out)
        G = G_root @ G_root.T

        natural_grad = [W]
        fisher_blocks = [(A, G)]

        # Override stubs
        evo._compute_advantages = lambda f: f # just pass through
        evo._compute_kfac_blocks = lambda pop: fisher_blocks
        evo._conjugate_gradient = lambda fb, adv: natural_grad

        # Capture update
        self.last_update = None
        def capture_update(update):
            self.last_update = update
        evo._apply_update = capture_update

        # Execute
        population = [nn.Linear(d_in, d_out)]
        fitnesses = [1.0]

        evo.trpo_step(population, fitnesses)

        # Verify
        # Calculate expected step size
        fisher_norm = torch.sum((W.T @ G @ W) * A).item()
        expected_step = math.sqrt(2 * 0.01 / fisher_norm)

        self.assertIsNotNone(self.last_update)
        self.assertEqual(len(self.last_update), 1)

        actual_update_tensor = self.last_update[0]
        expected_update_tensor = W * expected_step

        self.assertTrue(torch.allclose(actual_update_tensor, expected_update_tensor))

    def test_trpo_step_stub_compatibility(self):
        # Ensure it still works with default stubs (returning floats)
        evo = HighDimEvolution()

        # Default stubs return natural_grad=0.0, fisher_blocks=[]
        # _line_search_kl_constraint returns 0.1
        # trpo_step returns _apply_update(0.0 * 0.1) = _apply_update(0.0)

        res = evo.trpo_step([], [])
        # _apply_update returns None
        self.assertIsNone(res)
