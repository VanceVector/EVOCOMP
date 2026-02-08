import os
import torch
from enum import Enum
from dataclasses import dataclass
from typing import Any, List, Union
from .manifold import EvoManifoldKernel
from .distributed_context import DistributedEvoContext

# Stub for AI-Scientist integration
class AIScientistAgent:
    def __init__(self, template_dir, model):
        pass

    def generate_idea(self, context):
        return "New Hypothesis"

    def write_paper(self, idea, results, template):
        return "Paper Content"

class AutomatedEvoComp:
    """
    EVOCOMP enhanced with AI-Scientist meta-controller.
    """
    def __init__(self, world_size=None, rank=None, redis_host=None):

        # Fallback to env vars
        if world_size is None:
            world_size = int(os.environ.get("WORLD_SIZE", 1))
        if rank is None:
            rank = int(os.environ.get("RANK", 0))
        if redis_host is None:
            redis_host = os.environ.get("REDIS_HOST", 'localhost')

        self.evocomp = DistributedEvoContext(world_size=world_size, rank=rank, redis_host=redis_host)
        self.ai_scientist = AIScientistAgent(
            template_dir="templates/evo_comp",
            model="claude-3-opus-20240229"
        )

    def _design_experiment(self, hypothesis):
        return "Experiment Spec"

    def _submit_to_arxiv(self, paper):
        print("Submitting to arXiv: ", paper[:20] + "...")

    def meta_scientific_loop(self, iterations: int = 100):
        """
        Automated closed-loop research.
        """
        for i in range(iterations):
            # AI-Scientist analyzes current archive
            hypothesis = self.ai_scientist.generate_idea(
                context={
                    "current_pareto_front": self.evocomp.archive.fitnesses,
                    "manifold_coverage": self.evocomp.manifold.coverage() if self.evocomp.manifold else 0,
                    "recent_failures": self.evocomp.failure_log
                }
            )

            # EVOCOMP validates experimentally
            experiment = self._design_experiment(hypothesis)
            results = self.evocomp.execute(experiment)

            # Automated write-up if significant
            if results.significance_p < 0.05:
                paper = self.ai_scientist.write_paper(
                    idea=hypothesis,
                    results=results,
                    template="neurips_2024"
                )
                self._submit_to_arxiv(paper)

class ResearchAction(Enum):
    EXPLORE_NOVEL = 1
    EXPLOIT_CURRENT = 2

@dataclass
class ResearchState:
    embedding: Any

class MetaScientistNiche(EvoManifoldKernel):
    """
    AI-Scientist as an evolved niche in the ensemble.
    """
    def __init__(self, dim=10, exploration_threshold=0.5):
        super().__init__(dim)
        self.exploration_threshold = exploration_threshold
        self.successful_experiments: Union[List[torch.Tensor], torch.Tensor] = []
        # Cache for successful_experiments conversion
        self._history_cache = None
        self._history_list_ref = None  # Reference to the list object
        self._history_len = -1

    def forward(self, research_state: ResearchState) -> ResearchAction:
        """
        Determines next experimental steps.
        """
        # Prepare history tensor
        if isinstance(self.successful_experiments, list):
            if not self.successful_experiments:
                return ResearchAction.EXPLORE_NOVEL

            current_len = len(self.successful_experiments)

            # Check cache validity (same list object, length >= cached length)
            # We use 'is' to check for object identity, avoiding ABA problems with id()
            is_same_list = (self._history_list_ref is self.successful_experiments)

            if is_same_list and current_len == self._history_len and self._history_cache is not None:
                # Cache hit: exact match
                history = self._history_cache
            elif is_same_list and current_len > self._history_len and self._history_cache is not None:
                # Optimized append: only process new elements
                new_items = self.successful_experiments[self._history_len:]
                new_tensors = []
                for t in new_items:
                    if not isinstance(t, torch.Tensor):
                        new_tensors.append(torch.tensor(t))
                    else:
                        new_tensors.append(t)

                # Stack new items and concatenate with cache
                new_block = torch.stack(new_tensors)

                # Ensure device alignment if needed (usually handled by cat but good to be safe)
                if self._history_cache.device != new_block.device:
                    new_block = new_block.to(self._history_cache.device)

                history = torch.cat([self._history_cache, new_block])

                # Update cache
                self._history_cache = history
                self._history_len = current_len
            else:
                # Full rebuild (new list, truncated list, or first run)
                tensors = []
                for t in self.successful_experiments:
                    if not isinstance(t, torch.Tensor):
                        tensors.append(torch.tensor(t))
                    else:
                        tensors.append(t)
                history = torch.stack(tensors)

                # Update cache
                self._history_cache = history
                self._history_list_ref = self.successful_experiments
                self._history_len = current_len
        else:
            history = self.successful_experiments
            # Invalidate list cache if switched to tensor
            self._history_cache = None
            self._history_list_ref = None
            self._history_len = -1

            if history.numel() == 0:
                return ResearchAction.EXPLORE_NOVEL

        # Prepare current query
        current = research_state.embedding
        if isinstance(current, list):
            current = torch.tensor(current)

        if current.dim() == 1:
            current = current.unsqueeze(0) # (1, dim)

        if history.dim() == 1:
            history = history.unsqueeze(0)

        # Device alignment
        if current.device != history.device:
            history = history.to(current.device)

        # Vectorized pairwise distance: (1, N)
        dists = torch.cdist(current, history)

        # Min distance to any successful experiment
        min_dist = torch.min(dists).item()

        # If distance > threshold, it is novel.
        if min_dist > self.exploration_threshold:
            return ResearchAction.EXPLORE_NOVEL
        else:
            return ResearchAction.EXPLOIT_CURRENT
