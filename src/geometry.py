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

        # Randomized SVD
        # 1. Random projection with oversampling
        oversample = 5
        l = min(k + oversample, n_features)
        Omega = torch.randn(n_features, l)

        Y = X @ Omega

        # 2. Power iteration
        for _ in range(self.power_iters):
            Y = X @ (X.T @ Y)
            Q, _ = torch.linalg.qr(Y)
            Y = Q # Renormalize to prevent overflow

        # 3. Form Q
        Q, _ = torch.linalg.qr(Y)

        # 4. Form B = Q^T X
        B = Q.T @ X

        # 5. SVD of small matrix B
        U_tilde, S, Vh = torch.linalg.svd(B, full_matrices=False)

        # 6. Approximated SVD of X: U = Q @ U_tilde, S=S, Vh=Vh

        return ManifoldGeometry(basis=Vh[:k], spectrum=S[:k])
