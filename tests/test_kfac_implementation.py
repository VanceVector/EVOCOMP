import unittest
import torch
import torch.nn as nn
from src.evolution import HighDimEvolution

class TestKFAC(unittest.TestCase):
    def test_kfac_linear(self):
        evo = HighDimEvolution()

        # Define a simple linear model
        model = nn.Sequential(nn.Linear(10, 5))
        population = [model]

        # Register hooks
        evo.enable_kfac(population)

        # Forward and Backward pass
        input_data = torch.randn(2, 10)
        output = model(input_data)
        loss = output.sum()
        loss.backward()

        # Compute KFAC blocks
        blocks = evo._compute_kfac_blocks(population)

        # Verify
        self.assertEqual(len(blocks), 1)
        A, G = blocks[0]

        # Shapes: A should be (11, 11) because of bias, G should be (5, 5)
        self.assertEqual(A.shape, (11, 11))
        self.assertEqual(G.shape, (5, 5))

        # Check that they are not zero
        self.assertTrue(torch.norm(A) > 0)
        self.assertTrue(torch.norm(G) > 0)

        # Test disable hooks
        evo.disable_kfac()
        # Hooks should be gone.
        self.assertEqual(len(evo._kfac_hooks), 0)

    def test_kfac_linear_no_bias(self):
        evo = HighDimEvolution()
        model = nn.Sequential(nn.Linear(10, 5, bias=False))
        population = [model]
        evo.enable_kfac(population)

        input_data = torch.randn(2, 10)
        output = model(input_data)
        loss = output.sum()
        loss.backward()

        blocks = evo._compute_kfac_blocks(population)
        A, G = blocks[0]

        # Shapes: A should be (10, 10), G should be (5, 5)
        self.assertEqual(A.shape, (10, 10))
        self.assertEqual(G.shape, (5, 5))

    def test_kfac_conv2d(self):
        evo = HighDimEvolution()

        # Define a simple conv model
        # Input: (1, 3, 5, 5)
        # Conv: 3 in, 2 out, kernel 3, stride 1, padding 1
        model = nn.Sequential(nn.Conv2d(3, 2, kernel_size=3, padding=1))
        population = [model]

        evo.enable_kfac(population)

        input_data = torch.randn(1, 3, 5, 5)
        output = model(input_data)
        loss = output.sum()
        loss.backward()

        blocks = evo._compute_kfac_blocks(population)

        self.assertEqual(len(blocks), 1)
        A, G = blocks[0]

        # A shape: (C_in * K * K + 1, C_in * K * K + 1) -> (28, 28)
        self.assertEqual(A.shape, (28, 28))

        # G shape: (C_out, C_out) -> (2, 2)
        self.assertEqual(G.shape, (2, 2))

    def test_kfac_population_aggregation(self):
        evo = HighDimEvolution()

        # Population of 2 models (bias=False for simpler math check)
        model1 = nn.Linear(2, 2, bias=False)
        model2 = nn.Linear(2, 2, bias=False)

        with torch.no_grad():
            model2.weight.copy_(model1.weight)

        population = [model1, model2]
        evo.enable_kfac(population)

        # Run model 1
        x1 = torch.ones(1, 2) # [1, 1]
        y1 = model1(x1)
        y1.sum().backward()
        # A1 = [[1, 1], [1, 1]]

        # Run model 2 with different input
        x2 = torch.zeros(1, 2) # [0, 0]
        y2 = model2(x2)
        y2.sum().backward()
        # A2 = [[0, 0], [0, 0]]

        blocks = evo._compute_kfac_blocks(population)
        A, G = blocks[0]

        # Average A should be 0.5 * (A1 + A2) = [[0.5, 0.5], [0.5, 0.5]]
        expected_A = torch.tensor([[0.5, 0.5], [0.5, 0.5]])
        self.assertTrue(torch.allclose(A, expected_A))

if __name__ == '__main__':
    unittest.main()
