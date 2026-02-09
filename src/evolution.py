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

    def _get_flat_grad(self, population: List[nn.Module], advantages: torch.Tensor) -> torch.Tensor:
        """
        Computes the weighted sum of gradients across the population.
        g = sum(adv_i * grad_i)
        """
        if not population:
            return torch.tensor([])

        first_model = population[0]
        device = advantages.device

        total_params = sum(p.numel() for p in first_model.parameters())
        if total_params == 0:
            return torch.tensor([])

        flat_grad = torch.zeros(total_params, device=device)

        for model, adv in zip(population, advantages):
            offset = 0
            for p in model.parameters():
                numel = p.numel()
                if p.grad is not None:
                    g_flat = p.grad.view(-1)
                    flat_grad[offset : offset + numel] += adv * g_flat
                offset += numel

        return flat_grad

    def _get_param_shapes(self, model: nn.Module) -> List[Tuple[str, torch.Size, int]]:
        shapes = []
        for name, p in model.named_parameters():
            shapes.append((name, p.shape, p.numel()))
        return shapes

    def _fisher_vector_product(self, vec: torch.Tensor, fisher_blocks: Dict, param_shapes: List) -> torch.Tensor:
        """
        Computes F * v where F ~ block diag(G_l (x) A_l).
        """
        output = []
        offset = 0
        idx = 0

        while idx < len(param_shapes):
            name, shape, numel = param_shapes[idx]
            v_param = vec[offset : offset + numel]

            if '.' in name:
                module_name, param_type = name.rsplit('.', 1)
            else:
                module_name = ""
                param_type = name

            if module_name in fisher_blocks and param_type == 'weight':
                A, G = fisher_blocks[module_name]

                has_bias = False
                if idx + 1 < len(param_shapes):
                    next_name, next_shape, next_numel = param_shapes[idx+1]
                    if next_name == f"{module_name}.bias":
                        has_bias = True

                W_v = v_param.view(shape)

                if has_bias:
                    bias_offset = offset + numel
                    bias_name, bias_shape, bias_numel = param_shapes[idx+1]
                    v_bias = vec[bias_offset : bias_offset + bias_numel]

                    W_aug = torch.cat([W_v, v_bias.view(-1, 1)], dim=1)

                    if G.shape[1] == W_aug.shape[0] and A.shape[0] == W_aug.shape[1]:
                        res = G @ W_aug @ A
                        output.append(res[:, :-1].flatten())
                        output.append(res[:, -1].flatten())
                    else:
                        output.append(v_param)
                        output.append(v_bias)

                    offset += numel + bias_numel
                    idx += 2
                else:
                    if G.shape[1] == W_v.shape[0] and A.shape[0] == W_v.shape[1] + 1:
                        zeros = torch.zeros(W_v.shape[0], 1, device=W_v.device)
                        W_aug = torch.cat([W_v, zeros], dim=1)
                        res = G @ W_aug @ A
                        output.append(res[:, :-1].flatten())
                    elif G.shape[1] == W_v.shape[0] and A.shape[0] == W_v.shape[1]:
                        res = G @ W_v @ A
                        output.append(res.flatten())
                    else:
                        output.append(v_param)
                    offset += numel
                    idx += 1
            elif module_name in fisher_blocks and param_type == 'bias':
                output.append(v_param)
                offset += numel
                idx += 1
            else:
                output.append(v_param)
                offset += numel
                idx += 1

        result = torch.cat(output)
        result += 1e-3 * vec # Damping
        return result

    def _conjugate_gradient(self, fisher_blocks, gradient, param_shapes, max_iters=20, tol=1e-10):
        """
        Solves Fx = g for x using Conjugate Gradient.
        """
        x = torch.zeros_like(gradient)
        r = gradient.clone()
        p = gradient.clone()
        r_dot_r = torch.dot(r, r)

        for i in range(max_iters):
            Fp = self._fisher_vector_product(p, fisher_blocks, param_shapes)
            p_dot_Fp = torch.dot(p, Fp)

            if p_dot_Fp <= 0:
                break

            alpha = r_dot_r / p_dot_Fp
            x += alpha * p
            r -= alpha * Fp

            new_r_dot_r = torch.dot(r, r)
            if new_r_dot_r < tol:
                break

            beta = new_r_dot_r / r_dot_r
            p = r + beta * p
            r_dot_r = new_r_dot_r

        return x

    def _line_search_kl_constraint(self, natural_grad, fisher_blocks, param_shapes, delta=0.01):
        """
        Calculates step size beta such that 0.5 * beta^2 * v^T F v <= delta
        """
        Fv = self._fisher_vector_product(natural_grad, fisher_blocks, param_shapes)
        vFv = torch.dot(natural_grad, Fv)

        if vFv <= 1e-8:
            return 0.1 # Default small step if FIM is degenerate

        # 0.5 * beta^2 * vFv = delta => beta = sqrt(2 * delta / vFv)
        return torch.sqrt(2 * delta / vFv)

    def _apply_flat_update(self, model: nn.Module, update_vec: torch.Tensor):
        offset = 0
        for p in model.parameters():
            numel = p.numel()
            delta = update_vec[offset : offset + numel].view(p.shape)
            with torch.no_grad():
                p.add_(delta)
            offset += numel

    def _apply_update(self, population: List[nn.Module], update: torch.Tensor):
        if update is None: return
        for model in population:
            self._apply_flat_update(model, update)

    def trpo_step(self, population: List[nn.Module], fitnesses: List[float]):
        if not population: return

        # Compute advantages
        advantages = self._compute_advantages(fitnesses)

        # KFAC blocks : Fisher \approx A \otimes G for each layer
        fisher_blocks = self._compute_kfac_blocks(population)

        # Helper data
        param_shapes = self._get_param_shapes(population[0])

        # Compute gradient
        gradient = self._get_flat_grad(population, advantages)
        if gradient.numel() == 0:
            return

        # Conjugate gradient solve (no explicit inverse)
        natural_grad = self._conjugate_gradient(fisher_blocks, gradient, param_shapes)

        # Trust region constraint : KL < epsilon
        step_size = self._line_search_kl_constraint(natural_grad, fisher_blocks, param_shapes)

        # Apply update
        self._apply_update(population, natural_grad * step_size)

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
