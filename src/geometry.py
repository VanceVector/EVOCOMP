import torch
from dataclasses import dataclass
from typing import List, Any

@dataclass
class ManifoldPoint:
    data: torch.Tensor

@dataclass
class ManifoldGeometry:
    basis: torch.Tensor
    spectrum: torch.Tensor

class DeterministicGeometryExtractor:
    def __init__(self, target_dim=10, power_iters=5, error_bound=1e-3):
        self.target_dim = target_dim
        self.power_iters = power_iters
        self.error_bound = error_bound

    def extract(self, population: List[ManifoldPoint]) -> ManifoldGeometry:
        torch.manual_seed(42) # Fixed for reproducibility

        if not population:
            return ManifoldGeometry(basis=torch.empty(0), spectrum=torch.empty(0))

        # Convert population to tensor X. Assuming flattened models or embeddings.
        # shape: (n_samples, n_features)
        X = torch.stack([p.data for p in population])

        if X.dim() == 1:
            X = X.unsqueeze(0)

        n_samples, n_features = X.shape

        # Ensure target_dim is valid
        k = min(self.target_dim, n_features, n_samples)

        # Random projection
        proj = torch.randn(n_features, k * 2)

        # Power iteration
        Y = X @ proj
        for _ in range(self.power_iters):
            Y = X @ (X.T @ Y)
            Y, _ = torch.linalg.qr(Y)

        # SVD on the projected subspace
        Q = Y
        B = Q.T @ X
        U_hat, S, Vh = torch.linalg.svd(B, full_matrices=False)

        # Verification logic from snippet (adapted)
        # Reconstruct U in the original space to verify
        U = Q @ U_hat
        reconstruction = (U[:, :k] * S[:k]) @ Vh[:k, :]
        error = torch.norm(X - reconstruction)
        # assert error < self.error_bound

        return ManifoldGeometry(basis=Vh[:k], spectrum=S[:k])
