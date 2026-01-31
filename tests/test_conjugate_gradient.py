
import pytest
import torch
import torch.nn as nn
from src.evolution import HighDimEvolution

def test_conjugate_gradient_identity():
    """
    Test CG with empty fisher blocks (Identity approximation).
    Should solve (I + damping*I) x = g => x = g / (1 + damping)
    """
    evo = HighDimEvolution()

    g = torch.randn(10)
    fisher_blocks = [] # Empty -> Identity
    damping = 1e-3

    x = evo._conjugate_gradient(fisher_blocks, g, damping=damping)

    expected = g / (1 + damping)

    assert torch.allclose(x, expected, atol=1e-5)

def test_conjugate_gradient_kfac():
    """
    Test CG with one KFAC block.
    """
    evo = HighDimEvolution()
    torch.manual_seed(42)

    d_in, d_out = 3, 2

    # A and G must be symmetric positive definite
    A = torch.randn(d_in, d_in)
    A = A @ A.T + 0.1 * torch.eye(d_in)

    G = torch.randn(d_out, d_out)
    G = G @ G.T + 0.1 * torch.eye(d_out)

    fisher_blocks = [(A, G)]

    # Construct full F = G (kron) A
    F = torch.kron(G, A)
    damping = 1e-3
    F_damped = F + damping * torch.eye(d_in * d_out)

    g = torch.randn(d_in * d_out)

    x_cg = evo._conjugate_gradient(fisher_blocks, g, damping=damping)
    x_exact = torch.linalg.solve(F_damped, g)

    assert torch.allclose(x_cg, x_exact, rtol=1e-4, atol=1e-5)

def test_conjugate_gradient_list_advantages():
    """
    Test that list input for advantages is handled.
    """
    evo = HighDimEvolution()
    g_list = [1.0, 2.0, 3.0]
    g_tensor = torch.tensor(g_list)
    fisher_blocks = []

    x = evo._conjugate_gradient(fisher_blocks, g_list)

    expected = g_tensor / (1 + 1e-3)

    assert torch.allclose(x, expected)

def test_fisher_vector_product_mismatch():
    """
    Test that size mismatch raises error.
    """
    evo = HighDimEvolution()
    d_in, d_out = 2, 2
    A = torch.eye(d_in)
    G = torch.eye(d_out)

    fisher_blocks = [(A, G)]
    # Expects 4 params. Provide 3.
    vec = torch.randn(3)

    with pytest.raises(ValueError, match="Vector size mismatch"):
        evo._fisher_vector_product(fisher_blocks, vec)
