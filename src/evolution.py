import torch
import torch.nn as nn
from typing import List, Tuple, Any

class HighDimEvolution:
    def _compute_advantages(self, fitnesses):
        # Stub
        return fitnesses

    def _compute_kfac_blocks(self, population):
        # Stub
        return []

    def _conjugate_gradient(self, fisher_blocks, advantages):
        # Stub
        return 0.0

    def _line_search_kl_constraint(self, natural_grad):
        # Stub
        return 0.1

    def _apply_update(self, update):
        # Stub
        return None

    def trpo_step(self, population: List[nn.Module], fitnesses: List[float]):
        # Compute advantages
        advantages = self._compute_advantages(fitnesses)

        # KFAC blocks : Fisher \approx A \otimes G for each layer
        fisher_blocks = self._compute_kfac_blocks(population)

        # Conjugate gradient solve (no explicit inverse)
        natural_grad = self._conjugate_gradient(fisher_blocks, advantages)

        # Trust region constraint : KL < epsilon
        step_size = self._line_search_kl_constraint(natural_grad)

        return self._apply_update(natural_grad * step_size)

def select_diverse_pair(context, archive: List[nn.Module]) -> Tuple[nn.Module, nn.Module]:
    """
    Selects models with maximum geodesic distance for exploration.
    """
    if len(archive) < 2:
        return (archive[0], archive[0]) if archive else (None, None)

    embeddings = [context.manifold.embed(m) for m in archive]

    # Optimize using vectorized operations if possible
    # Assuming geodesic_distance corresponds to Euclidean distance in the manifold embedding space
    if isinstance(embeddings[0], torch.Tensor):
        embeddings_tensor = torch.stack(embeddings)
    else:
        embeddings_tensor = torch.tensor(embeddings)

    # Compute pairwise distances efficiently
    dists = torch.cdist(embeddings_tensor, embeddings_tensor, p=2)

    # Mask lower triangle and diagonal to ignore self-loops and duplicates
    mask = torch.triu(torch.ones_like(dists), diagonal=1)
    masked_dists = dists * mask

    # If all distances are zero (e.g. only 2 identical items or trivial manifold), return first pair
    if masked_dists.max() == 0:
        return (archive[0], archive[1])

    idx = torch.argmax(masked_dists)

    row = idx // dists.size(1)
    col = idx % dists.size(1)

    return (archive[int(row)], archive[int(col)])
