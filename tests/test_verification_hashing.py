import sys
import unittest
from unittest.mock import MagicMock
import hashlib

# Mock z3 if missing
try:
    import z3
    # Check if it's the real z3 or already a mock (e.g. from previous tests in same session, unlikely but possible)
    if isinstance(z3, MagicMock):
         Z3_AVAILABLE = False
    else:
         Z3_AVAILABLE = True
except ImportError:
    Z3_AVAILABLE = False
    z3 = MagicMock()
    sys.modules["z3"] = z3

# Mock torch if missing
try:
    import torch
except ImportError:
    torch = MagicMock()
    sys.modules["torch"] = torch
    # We also need to mock torch.nn for src/manifold.py
    torch_nn = MagicMock()
    sys.modules["torch.nn"] = torch_nn
    torch.nn = torch_nn
    # Ensure torch.nn.Module exists as it's used as base class
    torch_nn.Module = MagicMock

from src.verification import GuardVerifier

class TestProofHashing(unittest.TestCase):
    def test_hash_proof(self):
        verifier = GuardVerifier()

        # We need a solver object. If Z3_AVAILABLE is true, we use real z3.Solver.
        # If not, we use a MagicMock.
        solver = z3.Solver()

        if Z3_AVAILABLE:
            try:
                x = z3.Int('x')
                solver.add(x > 0)
                expected_smt2 = solver.to_smt2()
            except Exception:
                 # Fallback if real z3 fails for some reason or is a weird mock
                 expected_smt2 = "(assert (> x 0))"
                 solver.to_smt2 = MagicMock(return_value=expected_smt2)
        else:
            # Setup the mock behavior
            expected_smt2 = "(assert (> x 0))"
            solver.to_smt2.return_value = expected_smt2

        expected_hash = hashlib.sha256(expected_smt2.encode('utf-8')).hexdigest()

        proof_hash = verifier._hash_proof(solver)
        self.assertEqual(proof_hash, expected_hash)

if __name__ == '__main__':
    unittest.main()
