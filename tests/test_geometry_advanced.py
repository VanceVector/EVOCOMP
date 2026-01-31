import torch
from src.geometry import DeterministicGeometryExtractor, ManifoldPoint
import time
import pytest

def test_randomized_svd_impl():
    torch.manual_seed(42)

    # Parameters
    n_samples = 100
    n_features = 1000
    target_dim = 10
    power_iters = 5

    # Create random data with low rank structure
    # Generate low rank matrix: A = U * S * Vh
    rank = 20
    U_true = torch.randn(n_samples, rank)
    U_true, _ = torch.linalg.qr(U_true)
    V_true = torch.randn(n_features, rank)
    V_true, _ = torch.linalg.qr(V_true)
    S_true = torch.linspace(10, 1, rank)

    X = (U_true * S_true) @ V_true.T

    # Exact SVD for ground truth
    _, S_exact, Vh_exact = torch.linalg.svd(X, full_matrices=False)

    # Run Extractor (now with randomized SVD)
    extractor = DeterministicGeometryExtractor(target_dim=target_dim, power_iters=power_iters)
    pop = [ManifoldPoint(data=X[i]) for i in range(n_samples)]

    print("Running extractor (randomized implementation)...")
    start = time.time()
    geom = extractor.extract(pop)
    end = time.time()
    print(f"Extraction time: {end - start:.4f}s")

    # Verify Spectrum
    S_computed = geom.spectrum
    # We only get top 'target_dim' components

    diff_s = torch.norm(S_exact[:target_dim] - S_computed)
    print(f"Spectrum difference norm: {diff_s.item()}")

    assert diff_s < 1e-1, "Singular values deviate too much"

    k = min(target_dim, n_samples, n_features)
    Vh_ex_k = Vh_exact[:k]
    Vh_computed = geom.basis

    # Check orthogonality of computed basis
    orth_check = torch.norm(Vh_computed @ Vh_computed.T - torch.eye(k))
    print(f"Orthogonality check (should be ~0): {orth_check.item()}")
    assert orth_check < 1e-5

    # Check cosine similarity of 1st PC
    cos_sim = torch.abs(torch.dot(Vh_exact[0], Vh_computed[0]))
    print(f"Cosine similarity of 1st PC: {cos_sim.item()}")
    assert cos_sim > 0.99, "First PC direction mismatch"
