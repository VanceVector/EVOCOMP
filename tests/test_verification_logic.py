import unittest
import z3
from src.verification import GuardVerifier, SafetySpec
from src.manifold import EvoManifoldKernel

class TestVerificationLogic(unittest.TestCase):
    def test_unsafe_region_empty(self):
        verifier = GuardVerifier()
        spec = SafetySpec() # No bounds
        # Mock coords
        coords = [z3.Real(f'c_{i}') for i in range(2)]
        unsafe = verifier._encode_unsafe_region(spec, coords)

        # Should be False (boolean value, not Z3 BoolRef)
        self.assertEqual(unsafe, False)

    def test_unsafe_region_bounds(self):
        verifier = GuardVerifier()
        # Safe if x in [0, 10]
        spec = SafetySpec(safe_bounds={0: (0.0, 10.0)})
        coords = [z3.Real('c_0'), z3.Real('c_1')]

        unsafe = verifier._encode_unsafe_region(spec, coords)

        # Solver check
        s = z3.Solver()
        s.add(unsafe)

        # Test point inside safe region (e.g. 5) -> unsafe should be False (unsatisfiable if we assert unsafe)
        # We assert coords[0] == 5.
        # If unsafe is true, then 5 is unsafe.
        # But 5 is safe. So unsafe should be false.

        s.push()
        s.add(coords[0] == 5)
        # Should be unsat because unsafe is false for 5
        self.assertEqual(s.check(), z3.unsat)
        s.pop()

        # Test point outside safe region (e.g. 11) -> unsafe should be True
        s.push()
        s.add(coords[0] == 11)
        # Should be sat because 11 is unsafe
        self.assertEqual(s.check(), z3.sat)
        s.pop()

        # Test point outside safe region (e.g. -1) -> unsafe should be True
        s.push()
        s.add(coords[0] == -1)
        self.assertEqual(s.check(), z3.sat)
        s.pop()

    def test_verify_integration(self):
        verifier = GuardVerifier()
        manifold = EvoManifoldKernel(dim=2)

        # Case 1: Safe (No bounds -> safe everywhere)
        spec = SafetySpec()
        cert = verifier.verify(manifold, spec)
        self.assertTrue(cert.valid)

        # Case 2: Unsafe exists (because guard is True and we have unsafe regions)
        # Spec: safe in [0, 10].
        # Unsafe region is (-inf, 0) U (10, inf).
        spec = SafetySpec(safe_bounds={0: (0.0, 10.0)})
        cert = verifier.verify(manifold, spec)
        self.assertFalse(cert.valid)
        self.assertIsNotNone(cert.counterexample)

if __name__ == '__main__':
    unittest.main()
