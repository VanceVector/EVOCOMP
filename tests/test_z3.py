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
        # In our stub, unsafe is False, so And(False, guard) is False -> Unsat -> Valid=True
        # Wait, if unsafe is False (meaning no unsafe region), then the condition "unsafe AND guard" is False.
        # "False" is unsatisfiable. So solver.check() should be unsat.
        # So it returns ProofCertificate(valid=True).
        self.assertTrue(cert.valid)
        self.assertIsNotNone(cert.proof_hash)

    def test_manifold_theory(self):
        manifold = EvoManifoldKernel(dim=2)
        solver = ManifoldTheory(manifold)
        # Test adding constraint
        # center would be needed, but let's just test instantiation and add
        solver.add_geodesic_constraint(None, None, 1.0)
        # d_0^2 + d_1^2 < 1.0
        # Check satisfiability
        # We need to give values to d_0, d_1 or check if there exist such values.
        # Since we only added one constraint, it should be sat.
        self.assertEqual(solver.check(), z3.sat)
