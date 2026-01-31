import unittest
import torch
from src.geometry import DeterministicGeometryExtractor, ManifoldPoint

class TestGeometryAdvanced(unittest.TestCase):
    def test_randomized_svd_accuracy(self):
        torch.manual_seed(0)
        # Create a low rank matrix (N=100, D=50)
        # We need N > target_dim and D > target_dim
        U = torch.randn(100, 5)
        # Orthogonalize U to have nice singular values? Not strictly needed for rank check.
        # But let's just make X = U @ V.
        V = torch.randn(5, 50)
        X = U @ V # Rank 5

        points = [ManifoldPoint(data=X[i]) for i in range(100)]

        # target_dim=5 should capture almost all energy
        extractor = DeterministicGeometryExtractor(target_dim=5, power_iters=10)
        geometry = extractor.extract(points)

        # Check spectrum
        # The top 5 singular values should match reasonably well with true SVD
        _, S_true, _ = torch.linalg.svd(X, full_matrices=False)
        S_est = geometry.spectrum

        # Relative error
        error = torch.norm(S_true[:5] - S_est) / torch.norm(S_true[:5])
        self.assertLess(error, 1e-2, f"Spectrum error {error} too high")

    def test_reproducibility(self):
        # Even with different external seed, the extractor sets manual_seed(42) internally
        torch.manual_seed(123)
        X = torch.randn(20, 10)
        points = [ManifoldPoint(data=X[i]) for i in range(20)]

        extractor1 = DeterministicGeometryExtractor(target_dim=5, power_iters=2)
        geom1 = extractor1.extract(points)

        # Change seed externally
        torch.manual_seed(999)
        extractor2 = DeterministicGeometryExtractor(target_dim=5, power_iters=2)
        geom2 = extractor2.extract(points)

        self.assertTrue(torch.allclose(geom1.spectrum, geom2.spectrum))
        self.assertTrue(torch.allclose(geom1.basis, geom2.basis))

if __name__ == '__main__':
    unittest.main()
