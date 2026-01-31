import unittest
from evocomp import EvolutionaryEngine

class TestAPI(unittest.TestCase):
    def test_evolutionary_engine(self):
        # Initialize evolutionary engine
        engine = EvolutionaryEngine(
            world_size=4,
            model_class="vit_base",
            population_size=128
        )

        # Run evolution
        results = engine.evolve(
            generations=10,
            target_accuracy=0.72,
            budget=30000
        )

        self.assertGreaterEqual(results.best_accuracy, 0.72)
        self.assertEqual(results.generations, 10)
