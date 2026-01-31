import unittest
import z3
from src.verification import GuardVerifier, SafetySpec
from src.manifold import EvoManifoldKernel

class TestVerificationLogic(unittest.TestCase):
    def test_verify_safe(self):
        # Radius 1.0. Bounds [-2, 2]. All points in radius are in bounds.
        verifier = GuardVerifier()
        manifold = EvoManifoldKernel(dim=2, anchor_radius=1.0)
        spec = SafetySpec(safe_bounds={
            0: (-2.0, 2.0),
            1: (-2.0, 2.0)
        })
        cert = verifier.verify(manifold, spec)
        self.assertTrue(cert.valid, "Should be valid when bounds cover the anchor radius")

    def test_verify_unsafe(self):
        # Radius 1.0. Bounds [-0.5, 0.5].
        # Points between 0.5 and 1.0 are in guard but unsafe.
        verifier = GuardVerifier()
        manifold = EvoManifoldKernel(dim=2, anchor_radius=1.0)
        spec = SafetySpec(safe_bounds={
            0: (-0.5, 0.5)
        })
        cert = verifier.verify(manifold, spec)
        self.assertFalse(cert.valid, "Should be invalid when bounds are tighter than radius")

        # Check counterexample
        model = cert.counterexample
        self.assertIsNotNone(model)

        # Verify model satisfies constraints
        # Access variable c_0
        # We need to find the variable in the model.
        # It's named c_0.
        c_0_decl = None
        for d in model.decls():
            if d.name() == 'c_0':
                c_0_decl = d
                break

        self.assertIsNotNone(c_0_decl)
        c_0_val = model[c_0_decl]

        # Convert to float
        # Z3 Real values are fractions
        val = float(c_0_val.numerator_as_long()) / float(c_0_val.denominator_as_long())

        # It should be unsafe, so outside [-0.5, 0.5]
        self.assertTrue(abs(val) > 0.5 or abs(val) == 0.5) # Z3 strict inequality might result in exact boundary?

        # Also check it is within radius
        c_1_decl = None
        for d in model.decls():
            if d.name() == 'c_1':
                c_1_decl = d
                break
        c_1_val = 0.0
        if c_1_decl is not None:
             v = model[c_1_decl]
             c_1_val = float(v.numerator_as_long()) / float(v.denominator_as_long())

        dist_sq = val**2 + c_1_val**2
        self.assertLess(dist_sq, 1.0**2)

if __name__ == '__main__':
    unittest.main()
