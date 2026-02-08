import sys
from unittest.mock import MagicMock

try:
    import z3
except ImportError:
    # Mock z3 if not available so we can import modules that depend on it
    sys.modules["z3"] = MagicMock()
