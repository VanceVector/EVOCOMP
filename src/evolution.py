import torch
import torch.nn as nn
from typing import List, Tuple, Any, Dict, Optional

class HighDimEvolution:
    def __init__(self):
        self._kfac_hooks = []

    def _compute_advantages(self, fitnesses: List[float]) -> torch.Tensor:
        """
        Computes normalized advantages (Z-score).
        """
        if not fitnesses:
             return torch.tensor([])

        fits = torch.tensor(fitnesses, dtype=torch.float32)
        if len(fits) < 2:
            return torch.zeros_like(fits)

        std, mean = torch.std_mean(fits)
        if std < 1e-8:
            return torch.zeros_like(fits)

        return (fits - mean) / std

    def enable_kfac(self, population: List[nn.Module]):
        """
        Registers forward and backward hooks for K-FAC.
        """
        self.disable_kfac() # Ensure clean state

        for model in population:
            for name, layer in model.named_modules():
                if isinstance(layer, (nn.Linear, nn.Conv2d)):
                    # Forward hook
                    h1 = layer.register_forward_hook(self._get_activation_hook)
                    # Backward hook
                    h2 = layer.register_full_backward_hook(self._get_grad_hook)
                    self._kfac_hooks.extend([h1, h2])

    def disable_kfac(self):
        """
        Removes all registered K-FAC hooks.
        """
        for h in self._kfac_hooks:
            h.remove()
        self._kfac_hooks = []

    @staticmethod
    def _get_activation_hook(module, input, output):
        if not hasattr(module, '_kfac_a'):
            module._kfac_a = []
        # Input is a tuple, take first element
        module._kfac_a.append(input[0].detach())

    @staticmethod
    def _get_grad_hook(module, grad_input, grad_output):
        if not hasattr(module, '_kfac_g'):
            module._kfac_g = []
        # grad_output is a tuple
        module._kfac_g.append(grad_output[0].detach())

    def _compute_kfac_blocks(self, population: List[nn.Module]) -> Dict[str, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Aggregates covariances of activations (A) and gradients (G) across the population.
        Fisher ~ A (x) G
        """
        layer_stats = {}

        for model in population:
            for name, layer in model.named_modules():
                if isinstance(layer, (nn.Linear, nn.Conv2d)):
                    if hasattr(layer, '_kfac_a') and hasattr(layer, '_kfac_g'):
                        if name not in layer_stats:
                            layer_stats[name] = {'A': [], 'G': []}
                        layer_stats[name]['A'].extend(layer._kfac_a)
                        layer_stats[name]['G'].extend(layer._kfac_g)

                        # Cleanup to prevent memory leaks if models are reused
                        layer._kfac_a = []
                        layer._kfac_g = []

        blocks = {}
        for name, stats in layer_stats.items():
            if not stats['A'] or not stats['G']:
                continue

            # Stack data: (N_total, features)
            # Flatten batch dims
            A_list = [x.view(x.shape[0], -1) for x in stats['A']]
            G_list = [x.view(x.shape[0], -1) for x in stats['G']]

            if not A_list: continue

            A_data = torch.cat(A_list, dim=0)
            G_data = torch.cat(G_list, dim=0)

            # Handle Bias: Append 1s to A
            # Assuming if A has N samples, we append a column of 1s
            ones = torch.ones(A_data.shape[0], 1, device=A_data.device)
            A_data = torch.cat([A_data, ones], dim=1)

            N = A_data.shape[0]
            if N == 0: continue

            # Compute Covariances
            A_cov = (A_data.T @ A_data) / N
            G_cov = (G_data.T @ G_data) / N

            blocks[name] = (A_cov, G_cov)

        return blocks

    def _conjugate_gradient(self, fisher_blocks, advantages):
        # Stub: Real CG implementation would involve solving linear system.
        # Here we return a dummy natural gradient (scalar) to satisfy the interface logic flow.
        # In a real system, this would return a dictionary of parameter updates or a flattened vector.
        return 0.1

    def _line_search_kl_constraint(self, natural_grad):
        # Analytical step size calculation (Stubbed logic based on description)
        # beta = sqrt(2 delta / (v^T F v))
        # Assuming delta=0.01 and we calculate v^T F v
        # Since natural_grad is a dummy 0.1 here, we return a fixed step size.
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

def select_diverse_pair(context, archive: List[nn.Module]) -> Tuple[Optional[nn.Module], Optional[nn.Module]]:
    """
    Selects models with maximum geodesic distance for exploration using vectorized operations.
    """
    if not archive or len(archive) < 2:
        return (archive[0], archive[0]) if archive else (None, None)

    # Vectorized embedding
    embeddings = torch.stack([context.manifold.embed(m) for m in archive])

    # Pairwise distances (N, N)
    dists = torch.cdist(embeddings, embeddings)

    # Find indices of max distance
    # We want max(dists).
    # Since diagonal is 0, max will be off-diagonal unless all are same.

    argmax = torch.argmax(dists)
    n = dists.shape[0]
    i = argmax // n
    j = argmax % n

    return (archive[i], archive[j])
