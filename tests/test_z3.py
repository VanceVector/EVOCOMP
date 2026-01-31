import unittest
import z3
from src.verification import GuardVerifier, SafetySpec
from src.manifold import EvoManifoldKernel, ManifoldTheory

class TestZ3(unittest.TestCase):
    def test_verifier(self):
        verifier = GuardVerifier()
        manifold = EvoManifoldKernel(dim=3)
        spec = SafetySpec()
        cert = verifier.verify(manifold, spec)
        # If no unsafe region, it is trivially safe.
        self.assertTrue(cert.valid)
        self.assertIsNotNone(cert.proof_hash)

    def test_manifold_theory(self):
        manifold = EvoManifoldKernel(dim=2)
        solver = ManifoldTheory(manifold)

        # Create constants of the ManifoldPoint sort
        p = z3.Const('p', solver.ManifoldPoint)
        c = z3.Const('c', solver.ManifoldPoint)

        solver.add_geodesic_constraint(p, c, 1.0)

        # Check satisfiability
        # There exist points p, c such that dist(p, c) < 1.0.
        self.assertEqual(solver.check(), z3.sat)
