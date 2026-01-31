import torch
from enum import Enum
from dataclasses import dataclass
from typing import Any, List
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
    def __init__(self):
        # Stub: assuming 1 node for simplicity here
        self.evocomp = DistributedEvoContext(world_size=1, rank=0, redis_host='localhost')
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
        self.successful_experiments = torch.empty((0, dim), dtype=torch.float32)
        self.manifold = self

    def add_successful_experiment(self, embedding):
        """Adds a successful experiment embedding."""
        if not torch.is_tensor(embedding):
            embedding = torch.tensor(embedding, dtype=torch.float32)

        # Ensure embedding is on the same device as the storage
        embedding = embedding.to(self.successful_experiments.device)

        # Ensure embedding is (1, dim) or (dim) -> (1, dim)
        if embedding.dim() == 1:
            embedding = embedding.unsqueeze(0)

        if embedding.size(1) != self.dim:
             raise ValueError(f"Embedding dimension mismatch: {embedding.size(1)} != {self.dim}")

        self.successful_experiments = torch.cat([self.successful_experiments, embedding], dim=0)

    def forward(self, research_state: ResearchState) -> ResearchAction:
        """
        Determines next experimental steps.
        """
        experiments = self.successful_experiments

        # Handle legacy list case if externally modified
        if isinstance(experiments, list):
            if not experiments:
                 return ResearchAction.EXPLORE_NOVEL
            experiments = torch.tensor(experiments, dtype=torch.float32)
        elif torch.is_tensor(experiments):
             if experiments.numel() == 0:
                  return ResearchAction.EXPLORE_NOVEL
        else:
             # Unknown type (numpy?), try to convert
             experiments = torch.tensor(experiments, dtype=torch.float32)
             if experiments.numel() == 0:
                  return ResearchAction.EXPLORE_NOVEL

        # Convert research_state.embedding
        current_emb = research_state.embedding
        if not torch.is_tensor(current_emb):
            current_emb = torch.tensor(current_emb, dtype=torch.float32)

        # Ensure device match
        current_emb = current_emb.to(experiments.device)

        if current_emb.dim() == 1:
            current_emb = current_emb.unsqueeze(0) # (1, dim)

        # Geodesic distance to successful prior experiments
        # Using torch.cdist for Euclidean distance optimization

        dists = torch.cdist(current_emb, experiments)
        min_dist = torch.min(dists).item()

        similarity = min_dist

        if similarity > self.exploration_threshold:
            return ResearchAction.EXPLORE_NOVEL
        else:
            return ResearchAction.EXPLOIT_CURRENT
