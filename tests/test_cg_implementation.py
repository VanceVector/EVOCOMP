
import unittest
import torch
import torch.nn as nn
from src.evolution import HighDimEvolution

class TestCGImplementation(unittest.TestCase):
    def setUp(self):
        self.evo = HighDimEvolution()

    def test_cg_solver(self):
        # Create a simple model
        model = nn.Sequential(
            nn.Linear(2, 1)
        )
        # Weights: 2 params. Bias: 1 param. Total 3.
        # population of 1
        with torch.no_grad():
            model[0].weight.data = torch.tensor([[1.0, 2.0]])
            model[0].bias.data = torch.tensor([0.0])

        population = [model]

        # Spoof gradients
        model[0].weight.grad = torch.tensor([[0.1, 0.2]])
        model[0].bias.grad = torch.tensor([0.3])

        advantages = torch.tensor([1.0]) # 1 model, adv=1

        # Spoof fisher blocks
        # layer 0
        # A_cov (in+1, in+1) -> (3, 3)
        # G_cov (out, out) -> (1, 1)
        A = torch.eye(3)
        G = torch.eye(1)
        fisher_blocks = {"0": (A, G)}

        # Helpers
        param_shapes = self.evo._get_param_shapes(model)
        gradient = self.evo._get_flat_grad(population, advantages)

        # expected: [0.1, 0.2, 0.3] flattened.
        expected_grad = torch.tensor([0.1, 0.2, 0.3])
        self.assertTrue(torch.allclose(gradient, expected_grad))

        # Test Product
        # F = I (since A=I, G=I) + damping I
        # Fv = (1 + 1e-3) v

        prod = self.evo._fisher_vector_product(gradient, fisher_blocks, param_shapes)
        expected_prod = gradient * (1 + 1e-3)

        self.assertTrue(torch.allclose(prod, expected_prod), "Fisher product mismatch")

        # Test CG
        # Solve Fx = g.
        # Since F = (1+lam) I, x = g / (1+lam)

        x = self.evo._conjugate_gradient(fisher_blocks, gradient, param_shapes)

        expected_x = gradient / (1 + 1e-3)

        self.assertTrue(torch.allclose(x, expected_x, atol=1e-5), "CG result mismatch")

    def test_trpo_step(self):
        # Create models
        model1 = nn.Sequential(nn.Linear(2, 1))
        with torch.no_grad():
            model1[0].weight.data = torch.tensor([[1.0, 2.0]])
            model1[0].bias.data = torch.tensor([0.0])

        model2 = nn.Sequential(nn.Linear(2, 1))
        with torch.no_grad():
            model2[0].weight.data = torch.tensor([[1.0, 2.0]])
            model2[0].bias.data = torch.tensor([0.0])

        # Spoof gradients
        model1[0].weight.grad = torch.tensor([[0.1, 0.2]])
        model1[0].bias.grad = torch.tensor([0.3])

        model2[0].weight.grad = torch.tensor([[-0.1, -0.2]])
        model2[0].bias.grad = torch.tensor([-0.3])

        # Spoof kfac hooks data manually so _compute_kfac_blocks doesn't return empty
        # We need to simulate _kfac_a and _kfac_g on modules
        # A needs to be (N, in)
        # G needs to be (N, out)
        model1[0]._kfac_a = [torch.randn(1, 2)]
        model1[0]._kfac_g = [torch.randn(1, 1)]
        model2[0]._kfac_a = [torch.randn(1, 2)]
        model2[0]._kfac_g = [torch.randn(1, 1)]

        pop = [model1, model2]
        fits = [10.0, 5.0]

        initial_w = model1[0].weight.data.clone()

        self.evo.trpo_step(pop, fits)

        final_w = model1[0].weight.data

        self.assertFalse(torch.allclose(initial_w, final_w), "Model weights should have changed")

if __name__ == "__main__":
    unittest.main()
