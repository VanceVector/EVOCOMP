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

    def test_manifold_datatypes(self):
        manifold = EvoManifoldKernel(dim=2)
        solver = ManifoldTheory(manifold)

        # Access the dynamically created datatypes
        PointSort = solver.PointSort
        mk_point = solver.mk_point
        accessors = solver.accessors

        # Create two symbolic points
        p1 = z3.Const('p1', PointSort)
        p2 = z3.Const('p2', PointSort)

        # Add constraint: dist(p1, p2) < 1.0 (squared < 1.0)
        solver.add_geodesic_constraint(p1, p2, 1.0)

        # Add constraints to force them to be far apart
        # p1 = (0, 0), p2 = (2, 0) => dist^2 = 4
        solver.add(accessors[0](p1) == 0.0)
        solver.add(accessors[1](p1) == 0.0)
        solver.add(accessors[0](p2) == 2.0)
        solver.add(accessors[1](p2) == 0.0)

        # Should be unsat
        self.assertEqual(solver.check(), z3.unsat)

        # Reset and try a satisfying case
        solver.reset()

        solver.add_geodesic_constraint(p1, p2, 3.0) # radius 3 => radius^2 = 9. 4 < 9 is True.
        solver.add(accessors[0](p1) == 0.0)
        solver.add(accessors[1](p1) == 0.0)
        solver.add(accessors[0](p2) == 2.0)
        solver.add(accessors[1](p2) == 0.0)

        self.assertEqual(solver.check(), z3.sat)
