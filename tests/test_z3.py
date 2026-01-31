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

class RestrictiveGuardVerifier(GuardVerifier):
    def _encode_guard_condition(self, manifold, coords):
        # Enforce guard: coords[0] must be within [-0.5, 0.5]
        # This is essentially "safe" if the safety spec requires [-1, 1]
        if len(coords) > 0:
            return z3.And(coords[0] >= -0.5, coords[0] <= 0.5)
        return True

class TestSafetySpec(unittest.TestCase):
    def test_bounds_spec_unsafe_with_open_guard(self):
        """Test that defined bounds create unsafe regions, which are detected if guard is open."""
        verifier = GuardVerifier()
        manifold = EvoManifoldKernel(dim=1)
        # Safe if x in [-1, 1]. Unsafe if x < -1 or x > 1.
        # Guard is True (all space).
        # Unsafe region (e.g. x=2) is reachable.
        spec = SafetySpec(safe_bounds={0: (-1.0, 1.0)})
        cert = verifier.verify(manifold, spec)
        self.assertFalse(cert.valid, "Spec with bounds should be invalid if guard allows everything")

    def test_bounds_spec_safe_with_restrictive_guard(self):
        """Test that if guard restricts state to safe region, it is valid."""
        verifier = RestrictiveGuardVerifier()
        manifold = EvoManifoldKernel(dim=1)
        # Safe if x in [-1, 1].
        spec = SafetySpec(safe_bounds={0: (-1.0, 1.0)})
        # Guard enforces x in [-0.5, 0.5].
        # [-0.5, 0.5] is a subset of [-1, 1].
        # So it is impossible to be in unsafe region (x < -1 or x > 1) AND guard region.
        cert = verifier.verify(manifold, spec)
        self.assertTrue(cert.valid, "Should be valid if guard restricts to safe zone")
