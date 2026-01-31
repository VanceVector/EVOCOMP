import torch
import torch.nn as nn
import sys
import os
import math

# Add src to path
sys.path.append(os.path.join(os.getcwd(), 'src'))

from evolution import HighDimEvolution

def test_compute_kfac_blocks():
    # Setup
    evo = HighDimEvolution()

    # Create a small population of models
    class SimpleModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc1 = nn.Linear(2, 2, bias=False)

        def forward(self, x):
            return self.fc1(x)

    pop_size = 2
    population = [SimpleModel() for _ in range(pop_size)]

    # Initialize with specific weights
    # Model 0: I
    # Model 1: 3*I
    w0 = torch.eye(2)
    w1 = 3 * torch.eye(2)

    with torch.no_grad():
        population[0].fc1.weight.copy_(w0)
        population[1].fc1.weight.copy_(w1)

    # Call method
    try:
        blocks = evo._compute_kfac_blocks(population)
        print("Blocks computed:", len(blocks))

        # We expect 1 block (1 layer, no bias)
        assert len(blocks) == 1

        A, G = blocks[0]

        print("A shape:", A.shape)
        print("G shape:", G.shape)

        assert A.shape == (2, 2)
        assert G.shape == (2, 2)

        # Expected values
        # s = 2
        # scale = 1/sqrt(2)
        # A = I * scale
        # G = I * scale

        expected_scale = 1.0 / math.sqrt(2)
        expected_A = torch.eye(2) * expected_scale

        print("A:\n", A)
        print("Expected A:\n", expected_A)

        if torch.allclose(A, expected_A, atol=1e-5):
            print("A matches expected value.")
        else:
            print("A does NOT match expected value.")
            exit(1)

        if torch.allclose(G, expected_A, atol=1e-5):
             print("G matches expected value.")
        else:
            print("G does NOT match expected value.")
            exit(1)

        print("Test Passed!")

    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        exit(1)

def test_compute_kfac_blocks_with_bias():
    print("\nTest with Bias:")
    evo = HighDimEvolution()

    class BiasModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc1 = nn.Linear(2, 2, bias=True)

    pop_size = 2
    population = [BiasModel() for _ in range(pop_size)]

    # Weights same as before
    w0 = torch.eye(2)
    w1 = 3 * torch.eye(2)

    # Biases:
    # b0 = [0, 0]
    # b1 = [2, 2]
    # Mean b = [1, 1]
    # Delta b0 = [-1, -1]
    # Delta b1 = [1, 1]
    b0 = torch.zeros(2)
    b1 = torch.ones(2) * 2

    with torch.no_grad():
        population[0].fc1.weight.copy_(w0)
        population[1].fc1.weight.copy_(w1)
        population[0].fc1.bias.copy_(b0)
        population[1].fc1.bias.copy_(b1)

    blocks = evo._compute_kfac_blocks(population)
    print("Blocks computed:", len(blocks))

    # Expect 2 blocks: 1 for weight (A, G), 1 for bias (1, G_b)
    assert len(blocks) == 2

    A_w, G_w = blocks[0]
    A_b, G_b = blocks[1]

    print("Bias Block A:", A_b)
    print("Bias Block G_b:\n", G_b)

    assert torch.is_tensor(A_b) and A_b.item() == 1.0

    # Check G_b
    # Delta b0 = [-1, -1]. Outer = [[1, 1], [1, 1]]
    # Delta b1 = [1, 1]. Outer = [[1, 1], [1, 1]]
    # Mean Outer = [[1, 1], [1, 1]]

    expected_G_b = torch.ones(2, 2)

    if torch.allclose(G_b, expected_G_b, atol=1e-5):
        print("G_b matches expected value.")
    else:
        print("G_b does NOT match expected value.")
        exit(1)

    print("Test with Bias Passed!")

if __name__ == "__main__":
    test_compute_kfac_blocks()
    test_compute_kfac_blocks_with_bias()
