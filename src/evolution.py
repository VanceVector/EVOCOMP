import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Any

class HighDimEvolution:
    def __init__(self):
        self._kfac_hooks = []

    def _compute_advantages(self, fitnesses):
        # Stub
        return fitnesses

    def enable_kfac(self, population: List[nn.Module]):
        """
        Registers forward and backward hooks on the population models to capture
        activations (A) and gradients (G) for K-FAC computation.
        """
        # Ensure _kfac_hooks exists (in case __init__ was skipped or overwritten if used as mixin)
        if not hasattr(self, '_kfac_hooks'):
             self._kfac_hooks = []

        def save_input(module, input, output):
            # Save input activations
            # input is a tuple
            if isinstance(input, tuple):
                module._kfac_a = input[0].detach()
            else:
                module._kfac_a = input.detach()

        def save_grad_output(module, grad_input, grad_output):
            # Save output gradients
            # grad_output is a tuple
            if isinstance(grad_output, tuple):
                module._kfac_g = grad_output[0].detach()
            else:
                module._kfac_g = grad_output.detach()

        for model in population:
            for module in model.modules():
                if isinstance(module, (nn.Linear, nn.Conv2d)):
                    self._kfac_hooks.append(module.register_forward_hook(save_input))
                    self._kfac_hooks.append(module.register_full_backward_hook(save_grad_output))

    def disable_kfac(self):
        """
        Removes registered KFAC hooks.
        """
        if hasattr(self, '_kfac_hooks'):
            for hook in self._kfac_hooks:
                hook.remove()
            self._kfac_hooks = []

    def _compute_kfac_blocks(self, population: List[nn.Module]):
        """
        Computes the K-FAC blocks (A, G) for each supported layer, aggregated over the population.
        Assumes that `enable_kfac` has been called and a forward/backward pass has occurred.
        If bias is present, appends 1 to the input activation.
        """
        blocks = []
        if not population:
            return blocks

        # Assume all models in population have the same structure.
        # We iterate over the first model to identify layers, then aggregate stats from all models.
        # Note: This assumes that the 'modules' iterator returns modules in the same order.

        # Collect all models' modules in parallel lists
        all_models_modules = [list(m.modules()) for m in population]
        # Transpose to iterate layer by layer across population
        # zip(*all_models_modules) yields tuples of (layer_0_model_0, layer_0_model_1, ...)

        for layer_instances in zip(*all_models_modules):
            # Check the type of the first instance (assuming homogeneity)
            layer_0 = layer_instances[0]

            if isinstance(layer_0, (nn.Linear, nn.Conv2d)):
                A_sum = None
                G_sum = None
                count = 0

                for layer in layer_instances:
                    if hasattr(layer, '_kfac_a') and hasattr(layer, '_kfac_g'):
                        a = layer._kfac_a
                        g = layer._kfac_g

                        cov_a = None
                        cov_g = None
                        n_samples = 0

                        # Process based on layer type
                        if isinstance(layer, nn.Linear):
                            # a: (Batch, In) -> A = a^T a
                            # g: (Batch, Out) -> G = g^T g
                            if a.dim() > 2: a = a.view(-1, a.size(-1))
                            if g.dim() > 2: g = g.view(-1, g.size(-1))

                            # Append 1 if bias is present
                            if layer.bias is not None:
                                a = torch.cat([a, torch.ones_like(a[:, :1])], dim=1)

                            cov_a = a.t() @ a
                            cov_g = g.t() @ g
                            n_samples = a.size(0)

                        elif isinstance(layer, nn.Conv2d):
                            # a: (Batch, C_in, H, W)
                            # g: (Batch, C_out, H', W')
                            try:
                                a_unfold = F.unfold(a, layer.kernel_size, layer.dilation, layer.padding, layer.stride)
                                # a_unfold: (N, Cin*K*K, L)
                                # Permute to (N, L, Cin*K*K) -> reshape to (N*L, Cin*K*K)
                                a_unfold = a_unfold.permute(0, 2, 1).reshape(-1, a_unfold.size(1))

                                # Append 1 if bias is present
                                if layer.bias is not None:
                                    a_unfold = torch.cat([a_unfold, torch.ones_like(a_unfold[:, :1])], dim=1)

                                cov_a = a_unfold.t() @ a_unfold
                                n_samples = a_unfold.size(0)

                                # Gradients: (N, Cout, H', W')
                                # Permute to (N, H', W', Cout) -> reshape to (N*H'*W', Cout)
                                g_reshaped = g.permute(0, 2, 3, 1).reshape(-1, g.size(1))
                                cov_g = g_reshaped.t() @ g_reshaped

                            except Exception:
                                # Skip if dimensions mismatch or calculation fails
                                continue

                        if cov_a is not None and cov_g is not None:
                            if A_sum is None:
                                A_sum = cov_a
                                G_sum = cov_g
                            else:
                                A_sum += cov_a
                                G_sum += cov_g

                            count += n_samples

                if A_sum is not None and G_sum is not None and count > 0:
                    # Normalize by total number of samples
                    blocks.append((A_sum / count, G_sum / count))

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
