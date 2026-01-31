import torch
import torch.nn as nn
from typing import List, Tuple, Any

class HighDimEvolution:
    def _compute_advantages(self, fitnesses):
        # Stub
        return fitnesses

    def _compute_kfac_blocks(self, population: List[nn.Module]) -> List[Tuple[torch.Tensor, torch.Tensor]]:
        if not population:
            return []

        blocks = []

        # Helper to get flat modules from all models
        def get_all_modules(models):
            return [list(m.modules()) for m in models]

        all_modules_lists = get_all_modules(population)

        # Iterate over layers (modules)
        for layer_instances in zip(*all_modules_lists):
            # layer_instances is a tuple of the same layer from each model
            ref_layer = layer_instances[0]

            if isinstance(ref_layer, (nn.Linear, nn.Conv2d)):
                # Extract weights
                weights = [l.weight.data for l in layer_instances]

                # Check device from first weight
                device = weights[0].device

                # Stack to shape (N, out, in) or (N, out, in, k, k)
                W_stack = torch.stack(weights)
                N = W_stack.shape[0]

                # Reshape if Conv2d: (N, out, in*k*k)
                if isinstance(ref_layer, nn.Conv2d):
                    out_channels = W_stack.shape[1]
                    W_stack = W_stack.view(N, out_channels, -1)

                # Compute mean
                W_mean = W_stack.mean(dim=0) # (out, in)

                # Deviations: (N, out, in)
                Delta_W = W_stack - W_mean.unsqueeze(0)

                # Delta_W: (N, out, in)
                # Delta_W^T (transpose last 2 dims): (N, in, out)
                Delta_W_T = Delta_W.transpose(1, 2)

                # A_hat = (1/N) * sum(Delta_W_i^T @ Delta_W_i)
                A_hat = torch.bmm(Delta_W_T, Delta_W).mean(dim=0)

                # G_hat = (1/N) * sum(Delta_W_i @ Delta_W_i^T)
                G_hat = torch.bmm(Delta_W, Delta_W_T).mean(dim=0)

                # Normalization
                # tr(A * G) = tr(A) * tr(G) = s
                # s = mean(|Delta_W|_F^2)
                s = (Delta_W ** 2).sum(dim=(1, 2)).mean()

                if s > 1e-8:
                    scale = 1.0 / torch.sqrt(s)
                    A = A_hat * scale
                    G = G_hat * scale
                else:
                    A = A_hat
                    G = G_hat

                blocks.append((A, G))

                # Handle Bias
                if ref_layer.bias is not None:
                    biases = [l.bias.data for l in layer_instances]
                    b_stack = torch.stack(biases) # (N, out)
                    b_mean = b_stack.mean(dim=0)
                    Delta_b = b_stack - b_mean.unsqueeze(0) # (N, out)

                    # Covariance of bias: (out, out)
                    G_b = torch.bmm(Delta_b.unsqueeze(2), Delta_b.unsqueeze(1)).mean(dim=0)

                    blocks.append((torch.tensor(1.0, device=device), G_b))

        return blocks

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

    max_dist = 0
    best_pair = (archive[0], archive[1])

    for i, emb_i in enumerate(embeddings):
        for j, emb_j in enumerate(embeddings[i+1:], i+1):
            dist = context.manifold.geodesic_distance(emb_i, emb_j)
            if dist > max_dist:
                max_dist = dist
                best_pair = (archive[i], archive[j])

    return best_pair
