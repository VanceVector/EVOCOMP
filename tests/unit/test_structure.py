import unittest
import torch
import torch.nn as nn
from evocomp.manifold import EvoManifoldKernel
from evocomp.manifold import DeterministicGeometryExtractor, ManifoldPoint
from evocomp.distributed import DistributedEvoContext
from evocomp.evolution import HighDimEvolution, select_diverse_pair
from evocomp.verification import GuardVerifier, SafetySpec
from evocomp.core import EthicalCircuitBreaker
from evocomp.core import EvoCompEconomics
from evocomp.core import EvoCompBackend
from evocomp.core import AutomatedEvoComp, MetaScientistNiche, ResearchState, ResearchAction

class TestStructure(unittest.TestCase):
    def test_manifold(self):
        k = EvoManifoldKernel()
        self.assertEqual(k.coverage(), 0.5)

    def test_geometry(self):
        extractor = DeterministicGeometryExtractor()
        pop = [ManifoldPoint(data=torch.randn(10)) for _ in range(5)]
        geom = extractor.extract(pop)
        self.assertIsNotNone(geom)

    def test_evolution(self):
        evo = HighDimEvolution()
        self.assertTrue(hasattr(evo, 'trpo_step'))

        ctx = type('Context', (), {'manifold': EvoManifoldKernel()})()
        archive = [nn.Linear(1, 1), nn.Linear(1, 1)]
        pair = select_diverse_pair(ctx, archive)
        self.assertEqual(len(pair), 2)

    def test_serving(self):
        breaker = EthicalCircuitBreaker()
        self.assertTrue(breaker.validate(nn.Linear(1, 1)))

    def test_economics(self):
        eco = EvoCompEconomics()
        self.assertGreater(eco.estimate_total_cost(), 0)

    def test_backend(self):
        backend = EvoCompBackend()
        self.assertTrue(hasattr(backend, 'compile'))

    def test_meta(self):
        meta = AutomatedEvoComp()
        self.assertTrue(hasattr(meta, 'meta_scientific_loop'))

        niche = MetaScientistNiche()
        self.assertTrue(hasattr(niche, 'forward'))

        state = ResearchState(embedding=[0.0]*10)
        action = niche.forward(state)
        self.assertIn(action, [ResearchAction.EXPLORE_NOVEL, ResearchAction.EXPLOIT_CURRENT])
