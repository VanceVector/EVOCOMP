import unittest
import torch
import numpy as np
from src.geometry import DeterministicGeometryExtractor, ManifoldPoint

class TestGeometryMath(unittest.TestCase):
    def test_randomized_svd_accuracy(self):
        torch.manual_seed(42)
        n_samples = 100
        n_features = 50
        rank = 10
        target_dim = 10

        # Generate low-rank matrix X
        U_true = torch.randn(n_samples, rank)
        U_true, _ = torch.linalg.qr(U_true)
        V_true = torch.randn(n_features, rank)
        V_true, _ = torch.linalg.qr(V_true)
        S_true = torch.linspace(10, 1, rank)

        X = (U_true * S_true) @ V_true.T

        # Create population
        population = [ManifoldPoint(data=X[i]) for i in range(n_samples)]

        extractor = DeterministicGeometryExtractor(target_dim=target_dim, power_iters=10)
        geometry = extractor.extract(population)

        # Ground truth SVD
        U_gt, S_gt, Vh_gt = torch.linalg.svd(X, full_matrices=False)

        # Compare singular values
        # The first few singular values should be very close
        print("\nSingular Values Comparison:")
        print("Ground Truth:", S_gt[:target_dim])
        print("Extractor:   ", geometry.spectrum)

        # Check relative error of singular values
        s_error = torch.norm(S_gt[:target_dim] - geometry.spectrum) / torch.norm(S_gt[:target_dim])
        print(f"Spectrum Relative Error: {s_error.item()}")
        self.assertLess(s_error, 0.05, "Singular values deviate too much")

        # Check reconstruction error
        # Reconstruct X using the extracted basis
        # X_approx = X @ V @ V.T (projection onto the subspace spanned by V)
        # Or better: U * S * Vh. But we only have Vh and S.
        # We can project X onto Vh: coeff = X @ Vh.T
        # Reconstruction = coeff @ Vh

        Vh_extracted = geometry.basis
        # Vh_extracted is (k, n_features)

        # Project X onto the extracted basis
        X_projected = X @ Vh_extracted.T @ Vh_extracted

        recon_error = torch.norm(X - X_projected)

        # Compare with optimal reconstruction error (using ground truth top k)
        X_best_approx = (U_gt[:, :target_dim] * S_gt[:target_dim]) @ Vh_gt[:target_dim, :]
        best_error = torch.norm(X - X_best_approx)

        print(f"Reconstruction Error: {recon_error.item()}")
        print(f"Optimal Error:        {best_error.item()}")

        # The reconstruction error should be close to optimal
        # Since X is rank 10 and target_dim is 10, error should be close to 0 (machine epsilon)
        # But due to randomness, it might be slightly higher.

        self.assertLess(recon_error, best_error * 1.5 + 1e-3, "Reconstruction error is significantly worse than optimal")

    def test_dimensions(self):
        n_samples = 20
        n_features = 100
        target_dim = 15

        X = torch.randn(n_samples, n_features)
        population = [ManifoldPoint(data=X[i]) for i in range(n_samples)]

        extractor = DeterministicGeometryExtractor(target_dim=target_dim)
        geometry = extractor.extract(population)

        k = min(target_dim, n_samples, n_features)
        self.assertEqual(geometry.basis.shape, (k, n_features))
        self.assertEqual(geometry.spectrum.shape, (k,))

if __name__ == '__main__':
    unittest.main()
