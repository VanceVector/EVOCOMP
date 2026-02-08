import sys
from unittest.mock import MagicMock

# Mock dependencies that might be missing
sys.modules["torch"] = MagicMock()
sys.modules["torch.nn"] = MagicMock()
sys.modules["torch.distributed"] = MagicMock()
sys.modules["z3"] = MagicMock()
sys.modules["redis"] = MagicMock()
sys.modules["numpy"] = MagicMock()
sys.modules["scipy"] = MagicMock()
sys.modules["psutil"] = MagicMock()
sys.modules["pynvml"] = MagicMock()

import unittest
import json
from unittest.mock import patch, MagicMock
import os
import urllib.request
import urllib.error

from src.meta import AIScientistAgent

class TestAIScientistIntegration(unittest.TestCase):

    def setUp(self):
        self.template_dir = "templates/evo_comp"
        self.model = "claude-3-opus-20240229"

    def test_init_without_api_key(self):
        # Ensure environment variable is not set
        with patch.dict(os.environ, {}, clear=True):
            agent = AIScientistAgent(self.template_dir, self.model)
            self.assertIsNone(agent.api_key)

            response = agent.generate_idea({"test": "context"})
            self.assertEqual(response, "Simulated AI Scientist Response (No API Key)")

    def test_generate_idea_with_api_key(self):
        with patch.dict(os.environ, {"ANTHROPIC_API_KEY": "test-key"}):
            agent = AIScientistAgent(self.template_dir, self.model)
            self.assertEqual(agent.api_key, "test-key")

            # Mock response content
            mock_response_data = {
                "content": [
                    {"type": "text", "text": "Proposed Hypothesis"}
                ]
            }
            mock_response_json = json.dumps(mock_response_data).encode('utf-8')

            # Create a mock response object
            mock_resp = MagicMock()
            mock_resp.read.return_value = mock_response_json
            mock_resp.status = 200
            mock_resp.__enter__.return_value = mock_resp

            with patch('urllib.request.urlopen', return_value=mock_resp) as mock_urlopen:
                context = {"pareto_front": [1.0, 0.5]}
                hypothesis = agent.generate_idea(context)

                self.assertEqual(hypothesis, "Proposed Hypothesis")

                # Verify request
                args, kwargs = mock_urlopen.call_args
                req = args[0]
                self.assertEqual(req.full_url, "https://api.anthropic.com/v1/messages")

                # Check headers
                headers = {k.lower(): v for k, v in req.headers.items()}
                self.assertEqual(headers.get('x-api-key'), "test-key")

                # Verify payload
                payload = json.loads(req.data.decode('utf-8'))
                self.assertEqual(payload['model'], self.model)
                self.assertTrue("pareto_front" in payload['messages'][0]['content'])

    def test_write_paper_with_api_key(self):
        with patch.dict(os.environ, {"ANTHROPIC_API_KEY": "test-key"}):
            agent = AIScientistAgent(self.template_dir, self.model)

            mock_response_data = {
                "content": [
                    {"type": "text", "text": "Generated Paper Content"}
                ]
            }
            mock_response_json = json.dumps(mock_response_data).encode('utf-8')

            mock_resp = MagicMock()
            mock_resp.read.return_value = mock_response_json
            mock_resp.status = 200
            mock_resp.__enter__.return_value = mock_resp

            with patch('urllib.request.urlopen', return_value=mock_resp):
                paper = agent.write_paper("Hypothesis A", {"accuracy": 0.95}, "neurips_2024")
                self.assertEqual(paper, "Generated Paper Content")

    def test_api_error_handling(self):
        with patch.dict(os.environ, {"ANTHROPIC_API_KEY": "test-key"}):
            agent = AIScientistAgent(self.template_dir, self.model)

            # Mock HTTPError
            with patch('urllib.request.urlopen', side_effect=urllib.error.HTTPError(
                url="http://url", code=401, msg="Unauthorized", hdrs={}, fp=None
            )):
                response = agent.generate_idea({})
                self.assertIn("Error calling AI Scientist API: 401 Unauthorized", response)

    def test_generic_exception_handling(self):
        with patch.dict(os.environ, {"ANTHROPIC_API_KEY": "test-key"}):
            agent = AIScientistAgent(self.template_dir, self.model)

            with patch('urllib.request.urlopen', side_effect=Exception("Connection refused")):
                response = agent.generate_idea({})
                self.assertIn("Error calling AI Scientist API: Connection refused", response)

if __name__ == '__main__':
    unittest.main()
