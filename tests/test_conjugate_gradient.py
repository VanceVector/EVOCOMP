
import unittest
import torch
import torch.nn as nn
from src.evolution import HighDimEvolution

class TestConjugateGradient(unittest.TestCase):
    def setUp(self):
        self.evolution = HighDimEvolution()

    def test_cg_identity(self):
        """Test CG when fisher_blocks is empty (Identity matrix implied)."""
        fisher_blocks = []
        # Create a simple gradient vector (list of tensors)
        grads = [torch.ones(2, 2) * 2.0]

        # Expected solution: x = I^-1 * g = g
        result = self.evolution._conjugate_gradient(fisher_blocks, grads)

        self.assertEqual(len(result), 1)
        self.assertTrue(torch.allclose(result[0], grads[0]))

    def test_cg_simple_block(self):
        """Test CG with one block F = A \otimes G."""
        # A: 2x2 identity, G: 2x2 diagonal
        A = torch.eye(2)
        G = torch.diag(torch.tensor([2.0, 4.0]))
        fisher_blocks = [(A, G)]

        # Gradient
        # layer weight W is 2x2.
        # F vec(W) = vec(G W A).
        # We want to solve F x = g.
        # Let true solution x be all ones.
        true_x = torch.ones(2, 2)

        # Compute g = F true_x = G true_x A
        # G is diag(2,4), true_x is ones. A is I.
        # G true_x = [[2, 2], [4, 4]]
        # g = [[2, 2], [4, 4]]
        g = G @ true_x @ A
        grads = [g]

        result = self.evolution._conjugate_gradient(fisher_blocks, grads)

        self.assertEqual(len(result), 1)
        # Check if result matches true_x
        self.assertTrue(torch.allclose(result[0], true_x, atol=1e-5))

    def test_cg_multiple_blocks(self):
        """Test CG with multiple blocks."""
        # Block 1
        A1 = torch.eye(1)
        G1 = torch.tensor([[2.0]])
        # Block 2
        A2 = torch.eye(1)
        G2 = torch.tensor([[3.0]])

        fisher_blocks = [(A1, G1), (A2, G2)]

        grads = [torch.tensor([[2.0]]), torch.tensor([[3.0]])]

        # Expected x1 = 2/2 = 1
        # Expected x2 = 3/3 = 1

        result = self.evolution._conjugate_gradient(fisher_blocks, grads)

        self.assertEqual(len(result), 2)
        self.assertTrue(torch.allclose(result[0], torch.tensor([[1.0]]), atol=1e-5))
        self.assertTrue(torch.allclose(result[1], torch.tensor([[1.0]]), atol=1e-5))

if __name__ == '__main__':
    unittest.main()
