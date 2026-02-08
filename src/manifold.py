import z3
import torch
import torch.nn as nn
from typing import List, Union

class EvoManifoldKernel:
    """
    Represents the manifold kernel for evolutionary operations.
    """
    def __init__(self, dim=10, anchor_radius=1.0):
        self.dim = dim
        self.manifold_dim = dim
        self.anchor_radius = anchor_radius

    def embed(self, model: nn.Module) -> torch.Tensor:
        """
        Embeds a model into the manifold using deterministic random projection.
        """
        params = []
        for p in model.parameters():
            params.append(p.view(-1))

        if not params:
            return torch.zeros(self.dim)

        flat = torch.cat(params)
        n_features = flat.numel()

        # Deterministic projection matrix
        # Use a local generator to ensure reproducibility without affecting global state
        # We attempt to create the projection on the same device as the input to avoid transfers.
        device = flat.device

        # Generate projection matrix (dim, n_features)
        # Note: For very large models, this is memory intensive.
        # A real production system would use sparse projections or structured matrices (e.g. Fastfood transform).
        # Here we follow the instruction for deterministic random projection.

        # To save memory, we can compute it in chunks if needed, but let's keep it simple for now.
        # If n_features is huge, this line will OOM.
        # But assuming reasonable model sizes for this environment.

        try:
            g = torch.Generator(device=device)
            g.manual_seed(42)
            proj = torch.randn(self.dim, n_features, generator=g, device=device) / (self.dim ** 0.5)
        except RuntimeError:
            # Fallback to CPU if OOM occurs on GPU
            g = torch.Generator(device='cpu')
            g.manual_seed(42)
            proj = torch.randn(self.dim, n_features, generator=g, device='cpu') / (self.dim ** 0.5)

        # Move to same device
        if flat.device != proj.device:
            # Usually keep proj on CPU to avoid VRAM usage, move flat to CPU
            flat = flat.to(proj.device)

        embedding = proj @ flat

        # Ensure embedding is returned on the original device
        # This handles the case where we fell back to CPU but the caller expects GPU
        if embedding.device != device:
            embedding = embedding.to(device)

        return embedding

    def geodesic_distance(self, emb_i: Union[torch.Tensor, List], emb_j: Union[torch.Tensor, List]) -> float:
        """
        Computes geodesic distance between two embeddings.
        """
        if isinstance(emb_i, list):
            emb_i = torch.tensor(emb_i)
        if isinstance(emb_j, list):
            emb_j = torch.tensor(emb_j)

        # Ensure same device
        if emb_i.device != emb_j.device:
            emb_j = emb_j.to(emb_i.device)

        return torch.norm(emb_i - emb_j).item()

    def coverage(self):
        """
        Returns the coverage metric of the manifold.
        """
        return 0.5

    def distance(self, emb_i, emb_j):
        return self.geodesic_distance(emb_i, emb_j)


class ManifoldTheory(z3.Solver):
    def __init__(self, manifold):
        super().__init__()
        self.manifold = manifold
        self.ManifoldPoint = None
        self.register_datatypes()

    def register_datatypes(self):
        # Create a Datatype for Point
        Point = z3.Datatype('ManifoldPoint')
        # Constructor 'point' with 'dim' reals
        args = [(f'x_{i}', z3.RealSort()) for i in range(self.manifold.dim)]
        Point.declare('point', *args)
        self.ManifoldPoint = Point.create()

    def add_geodesic_constraint(self, point, center, radius):
        """
        Encodes: distance_manifold(point, center) < radius
        point and center are Z3 expressions of Sort ManifoldPoint
        """
        dist_sq = 0
        for i in range(self.manifold.dim):
            # Accessor for dimension i
            # The constructor is the first (and only) one, index 0.
            # The accessor for field i is at index i.
            acc = self.ManifoldPoint.accessor(0, i)

            p_val = acc(point)
            c_val = acc(center)
            dist_sq += (p_val - c_val) ** 2

        self.add(dist_sq < radius**2)
