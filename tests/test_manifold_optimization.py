import unittest
from unittest.mock import MagicMock
import sys

# Mock z3 and torch before importing src.manifold
sys.modules['z3'] = MagicMock()
sys.modules['torch'] = MagicMock()
sys.modules['torch.nn'] = MagicMock()

import torch
from src.manifold import EvoManifoldKernel

class TestManifoldOptimization(unittest.TestCase):
    def setUp(self):
        # Reset mocks
        torch.reset_mock()

        # Setup common mock behaviors
        self.flat_tensor = MagicMock()
        self.flat_tensor.numel.return_value = 100
        torch.cat.return_value = self.flat_tensor

        # Mock parameters
        self.model = MagicMock()
        p = MagicMock()
        self.model.parameters.return_value = [p]
        p.view.return_value = p

    def test_embed_cpu_device(self):
        """Test embedding when model is on CPU."""
        # Setup flat tensor device
        self.flat_tensor.device = 'cpu'

        # Setup Generator and randn
        g_mock = MagicMock()
        torch.Generator.return_value = g_mock
        proj_mock = MagicMock()
        proj_mock.device = 'cpu'
        torch.randn.return_value = proj_mock

        kernel = EvoManifoldKernel(dim=10)
        kernel.embed(self.model)

        # Verify Generator created with correct device
        torch.Generator.assert_called_with(device='cpu')

        # Verify randn called with correct generator and device
        torch.randn.assert_called_with(10, 100, generator=g_mock, device='cpu')

    def test_embed_cuda_device(self):
        """Test embedding when model is on CUDA."""
        # Setup flat tensor device
        self.flat_tensor.device = 'cuda:0'
        self.flat_tensor.is_cuda = True

        # Setup Generator and randn
        g_mock = MagicMock()
        torch.Generator.return_value = g_mock
        proj_mock = MagicMock()
        proj_mock.device = 'cuda:0'
        torch.randn.return_value = proj_mock

        kernel = EvoManifoldKernel(dim=10)
        kernel.embed(self.model)

        # Verify Generator created with correct device
        torch.Generator.assert_called_with(device='cuda:0')

        # Verify randn called with correct generator and device
        torch.randn.assert_called_with(10, 100, generator=g_mock, device='cuda:0')

    def test_embed_cuda_oom_fallback(self):
        """Test fallback to CPU when CUDA OOM occurs."""
        # Setup flat tensor device
        self.flat_tensor.device = 'cuda:0'

        # Setup Generator to return mock
        g_mock = MagicMock()
        torch.Generator.return_value = g_mock

        # Setup explicit mocks to track operations
        cpu_proj_mock = MagicMock()
        cpu_proj_mock.device = 'cpu'
        # Division result
        proj_div_mock = MagicMock()
        proj_div_mock.device = 'cpu'
        cpu_proj_mock.__truediv__.return_value = proj_div_mock

        # Matmul result (embedding)
        embedding_mock = MagicMock()
        embedding_mock.device = 'cpu'
        proj_div_mock.__matmul__.return_value = embedding_mock

        # Setup randn to raise RuntimeError first (OOM), then return CPU tensor
        def side_effect(*args, **kwargs):
            device = kwargs.get('device')
            if device == 'cuda:0':
                raise RuntimeError("CUDA out of memory")
            return cpu_proj_mock

        torch.randn.side_effect = side_effect

        kernel = EvoManifoldKernel(dim=10)
        kernel.embed(self.model)

        # Verify Generator called with cuda first
        # torch.Generator.assert_any_call(device='cuda:0')
        # Note: Depending on implementation, might call Generator('cuda:0') then Generator('cpu')

        # Verify randn called with cuda, then cpu
        # check call args list
        calls = torch.randn.call_args_list
        # Should be at least 2 calls
        self.assertGreaterEqual(len(calls), 2)

        # First call should be cuda
        self.assertEqual(calls[0].kwargs['device'], 'cuda:0')

        # Verify flat tensor moved to CPU
        # self.flat_tensor.cpu.assert_called()
        # Implementation changed to use .to(device)
        self.flat_tensor.to.assert_called_with('cpu')

        # Verify embedding moved back to original device (cuda:0)
        embedding_mock.to.assert_called_with('cuda:0')

if __name__ == '__main__':
    unittest.main()
