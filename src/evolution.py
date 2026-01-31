import math
import torch
import torch.nn as nn
from typing import List, Tuple, Any, Union

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

    def _line_search_kl_constraint(self, natural_grad: Union[List[torch.Tensor], float], fisher_blocks: List[Tuple[torch.Tensor, torch.Tensor]] = None, max_kl: float = 0.01) -> float:
        """
        Calculates the step size to satisfy the KL constraint using the Fisher Information Matrix.
        Uses K-FAC approximation: F ~ A x G
        """
        if isinstance(natural_grad, float) or not fisher_blocks:
            return 0.1

        fisher_norm = 0.0

        if len(natural_grad) != len(fisher_blocks):
            return 0.1

        for i, update in enumerate(natural_grad):
            A, G = fisher_blocks[i]

            if update.dim() == 2:
                # Tr(W^T G W A) = sum((W^T G W) * A)
                term = torch.sum((update.T @ G @ update) * A)
            elif update.dim() == 1:
                # For bias/vectors: v^T G v * A (if A is scalar factor)
                # Assuming A matches input dim, for bias input is 1.
                term = update @ G @ update
                if A.numel() == 1:
                    term = term * A.item()
            else:
                term = 0.0

            fisher_norm += term

        if isinstance(fisher_norm, torch.Tensor):
            fisher_norm = fisher_norm.item()

        if fisher_norm <= 1e-8:
            return 1.0

        step_size = math.sqrt(2 * max_kl / fisher_norm)
        return float(step_size)

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
        step_size = self._line_search_kl_constraint(natural_grad, fisher_blocks)

        if isinstance(natural_grad, list):
            scaled_update = [g * step_size for g in natural_grad]
        else:
            scaled_update = natural_grad * step_size

        return self._apply_update(scaled_update)

def select_diverse_pair(context, archive: List[nn.Module]) -> Tuple[nn.Module, nn.Module]:
    """
    Selects models with maximum geodesic distance for exploration.
    """
    if len(archive) < 2:
        return (archive[0], archive[0]) if archive else (None, None)

    embeddings = [context.manifold.embed(m) for m in archive]

    max_dist = 0
    best_pair = (archive[0], archive[1])

    for i, emb_i in enumerate(embeddings):
        for j, emb_j in enumerate(embeddings[i+1:], i+1):
            dist = context.manifold.geodesic_distance(emb_i, emb_j)
            if dist > max_dist:
                max_dist = dist
                best_pair = (archive[i], archive[j])

    return best_pair
