import unittest
import z3
from src.verification import GuardVerifier, SafetySpec
from src.manifold import EvoManifoldKernel

class TestVerifier(GuardVerifier):
    def __init__(self, unsafe_cond_func):
        self.unsafe_cond_func = unsafe_cond_func

    def _encode_unsafe_region(self, spec, coords):
        return self.unsafe_cond_func(coords)

class TestGuardLogic(unittest.TestCase):
    def test_overlap(self):
        """
        Test that when unsafe region overlaps with guard region, verification fails (valid=False).
        """
        manifold = EvoManifoldKernel(dim=2, anchor_radius=1.0)
        # Unsafe region: x > 0.5.
        # Guard region: x^2 + y^2 <= 1.0.
        # Intersection exists (e.g., x=0.6, y=0).

        def unsafe_logic(coords):
            return coords[0] > 0.5

        verifier = TestVerifier(unsafe_logic)
        spec = SafetySpec()

        cert = verifier.verify(manifold, spec)
        self.assertFalse(cert.valid)
        self.assertIsNotNone(cert.counterexample)

        # Check counterexample
        model = cert.counterexample
        # Model should satisfy: x > 0.5 AND x^2 + y^2 <= 1.0
        # We need to retrieve values from the model.
        # coords are created inside verify method, so we can't easily access them here to evaluate against model directly
        # unless we reconstruct the variables with same names.

        c_0 = z3.Real('c_0')
        c_1 = z3.Real('c_1')

        val_0 = model.eval(c_0)
        val_1 = model.eval(c_1)

        # Z3 values can be fractions. Convert to float for check.
        def to_float(v):
            if z3.is_rational_value(v):
                return float(v.numerator_as_long()) / float(v.denominator_as_long())
            if z3.is_algebraic_value(v):
                 return float(v.approx(10).numerator_as_long()) / float(v.approx(10).denominator_as_long()) # approximation
            return float(v.as_decimal(10).replace('?', ''))

        f_0 = to_float(val_0)
        f_1 = to_float(val_1)

        self.assertGreater(f_0, 0.5)
        self.assertLessEqual(f_0**2 + f_1**2, 1.0 + 1e-9) # tolerance

    def test_no_overlap(self):
        """
        Test that when unsafe region does not overlap with guard region, verification passes (valid=True).
        """
        manifold = EvoManifoldKernel(dim=2, anchor_radius=1.0)
        # Unsafe region: x > 1.5.
        # Guard region: x^2 + y^2 <= 1.0.
        # No intersection.

        def unsafe_logic(coords):
            return coords[0] > 1.5

        verifier = TestVerifier(unsafe_logic)
        spec = SafetySpec()

        cert = verifier.verify(manifold, spec)
        self.assertTrue(cert.valid)
        self.assertIsNone(cert.counterexample)

if __name__ == '__main__':
    unittest.main()
