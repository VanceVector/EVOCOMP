import os
import torch
import json
import urllib.request
import urllib.error
from enum import Enum
from dataclasses import dataclass
from typing import Any, List, Union
from .manifold import EvoManifoldKernel
from .distributed_context import DistributedEvoContext

# Stub for AI-Scientist integration
class AIScientistAgent:
    def __init__(self, template_dir, model="claude-3-opus-20240229"):
        self.template_dir = template_dir
        self.model = model
        self.api_key = os.environ.get("ANTHROPIC_API_KEY")
        self.api_url = "https://api.anthropic.com/v1/messages"

    def _call_anthropic_api(self, system_prompt: str, user_prompt: str) -> str:
        """
        Helper method to call Anthropic API.
        """
        if not self.api_key:
            # Fallback/Mock behavior if no API key is present
            return "Simulated AI Scientist Response (No API Key)"

        headers = {
            "x-api-key": self.api_key,
            "anthropic-version": "2023-06-01",
            "content-type": "application/json"
        }

        payload = {
            "model": self.model,
            "max_tokens": 4096,
            "system": system_prompt,
            "messages": [
                {"role": "user", "content": user_prompt}
            ]
        }

        try:
            req = urllib.request.Request(
                self.api_url,
                data=json.dumps(payload).encode('utf-8'),
                headers=headers,
                method='POST'
            )

            with urllib.request.urlopen(req) as response:
                if response.status == 200:
                    result = json.loads(response.read().decode('utf-8'))
                    # Anthropic response format: {"content": [{"text": "...", "type": "text"}], ...}
                    content_blocks = result.get("content", [])
                    if content_blocks and content_blocks[0].get("type") == "text":
                        return content_blocks[0].get("text", "")
                    return ""
                else:
                    return f"Error: API returned status {response.status}"

        except urllib.error.HTTPError as e:
            return f"Error calling AI Scientist API: {e.code} {e.reason}"
        except Exception as e:
            return f"Error calling AI Scientist API: {str(e)}"

    def generate_idea(self, context):
        system_prompt = "You are an AI Scientist. Analyze the provided context and generate a novel scientific hypothesis for evolutionary computation."

        # Format context safely
        try:
            context_str = json.dumps(context, indent=2, default=str)
        except Exception:
            context_str = str(context)

        user_prompt = f"Here is the current experimental context:\n{context_str}\n\nPlease propose a new hypothesis to test."

        return self._call_anthropic_api(system_prompt, user_prompt)

    def write_paper(self, idea, results, template):
        system_prompt = "You are an AI Scientist. Write a scientific paper based on the provided hypothesis and experimental results."

        results_str = str(results)

        user_prompt = f"Hypothesis: {idea}\n\nResults: {results_str}\n\nTemplate Style: {template}\n\nPlease write the paper content."

        return self._call_anthropic_api(system_prompt, user_prompt)

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

    def forward(self, research_state: ResearchState) -> ResearchAction:
        """
        Determines next experimental steps.
        """
        # Prepare history tensor
        if isinstance(self.successful_experiments, list):
            if not self.successful_experiments:
                return ResearchAction.EXPLORE_NOVEL
            # Ensure elements are tensors
            tensors = []
            for t in self.successful_experiments:
                if not isinstance(t, torch.Tensor):
                    tensors.append(torch.tensor(t))
                else:
                    tensors.append(t)
            history = torch.stack(tensors)
        else:
            history = self.successful_experiments
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
