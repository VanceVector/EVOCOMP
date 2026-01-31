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

    def _fisher_vector_product(self, fisher_blocks, vec):
        """
        Computes F * vec where F is approximated by KFAC blocks.
        fisher_blocks: List of (A, G) tuples.
        vec: Flattened parameter vector.
        """
        result = []
        idx = 0
        for A, G in fisher_blocks:
            # A: (d_in, d_in), G: (d_out, d_out)
            # Param shape corresponding to this block: (d_out, d_in)
            d_out = G.shape[0]
            d_in = A.shape[0]
            num_params = d_out * d_in

            # Extract segment from vec
            if idx + num_params > vec.numel():
                raise ValueError(f"Vector size mismatch. Expected at least {idx + num_params}, got {vec.numel()}")

            v_block = vec[idx : idx + num_params]
            idx += num_params

            # Reshape to (d_out, d_in)
            v_matrix = v_block.view(d_out, d_in)

            # Compute G * V * A
            # Note: A and G are symmetric covariances
            res_matrix = torch.matmul(torch.matmul(G, v_matrix), A)

            # Flatten and append
            result.append(res_matrix.view(-1))

        # If there are remaining elements in vec that were not covered by blocks,
        # we treat them as identity (or just return them as is, assuming independent/diagonal=1).
        if idx < vec.numel():
             result.append(vec[idx:])

        return torch.cat(result)

    def _conjugate_gradient(self, fisher_blocks, advantages, max_iters=20, tol=1e-10, damping=1e-3):
        """
        Solves F * x = g using Conjugate Gradient.
        fisher_blocks: List of KFAC blocks.
        advantages: Gradient vector g (flattened).
        """
        # Ensure advantages is a tensor
        if isinstance(advantages, list):
             advantages = torch.tensor(advantages, dtype=torch.float32)

        b = advantages
        x = torch.zeros_like(b)
        r = b.clone()
        p = r.clone()

        rdotr = torch.dot(r, r)

        for i in range(max_iters):
            # Compute F * p
            z = self._fisher_vector_product(fisher_blocks, p)

            # Apply damping: (F + lambda I) * p = Fp + lambda * p
            z = z + damping * p

            p_dot_z = torch.dot(p, z)

            if p_dot_z <= 1e-20:
                break

            alpha = rdotr / p_dot_z
            x = x + alpha * p
            r = r - alpha * z

            new_rdotr = torch.dot(r, r)
            if new_rdotr < tol:
                break

            beta = new_rdotr / rdotr
            p = r + beta * p
            rdotr = new_rdotr

        return x

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
