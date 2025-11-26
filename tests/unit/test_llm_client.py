import unittest
from unittest.mock import MagicMock, patch
import sys
import os
import json
from omegaconf import OmegaConf

# Add the src directory to the system path to allow imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../src')))

from ctr_labeling.llm_client import LLMClient

class TestLLMClient(unittest.TestCase):
    def setUp(self):
        """Set up a dummy configuration for testing."""
        self.cfg = OmegaConf.create({
            "api": {
                "api_key": "test-key",
                "model": "gpt-test",
                "temperature": 0.0,
                "max_retries": 1
            },
            "prompt": {
                "system_prompt": "You are a labeler.",
                "labels": ["Label_A", "Label_B"]
            }
        })
        self.empty_labels = {"Label_A": 0, "Label_B": 0}
        self.empty_meta = {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
            "latency_seconds": 0.0,
            "status": "skipped",
            "error_message": "",
            "request_id": "",
            "model_version": "",
            "retry_count": 0,
            "started_at_utc": "",
            "ended_at_utc": ""
        }

    def test_get_labels_empty_report(self):
        """Test that an empty report immediately returns all zeros and empty meta without calling the API."""
        with patch('ctr_labeling.llm_client.OpenAI') as mock_openai:
            client = LLMClient(self.cfg)
            
            # Test None
            labels_none, meta_none = client.get_labels(None)
            self.assertEqual(labels_none, self.empty_labels)
            self.assertEqual(meta_none, self.empty_meta)
            
            # Test Empty String
            labels_empty, meta_empty = client.get_labels("")
            self.assertEqual(labels_empty, self.empty_labels)
            self.assertEqual(meta_empty, self.empty_meta)
            
            # Verify API was never called
            mock_instance = mock_openai.return_value
            mock_instance.chat.completions.create.assert_not_called()

    @patch('ctr_labeling.llm_client.OpenAI')
    def test_get_labels_success(self, mock_openai):
        """Test a successful API response with valid JSON and token usage."""
        # Setup the mock response
        mock_response = MagicMock()
        mock_response.choices[0].message.content = json.dumps({
            "Label_A": 1,
            "Label_B": 0
        })
        # Setup token usage
        mock_response.usage.prompt_tokens = 150
        mock_response.usage.completion_tokens = 10
        mock_response.usage.total_tokens = 160
        mock_response.id = "resp_123"
        mock_response.model = "gpt-test-2025"
        
        mock_instance = mock_openai.return_value
        mock_instance.chat.completions.create.return_value = mock_response
        
        client = LLMClient(self.cfg)
        labels, meta = client.get_labels("Patient has condition A.")
        
        expected_labels = {"Label_A": 1, "Label_B": 0}
        
        self.assertEqual(labels, expected_labels)
        self.assertEqual(meta["prompt_tokens"], 150)
        self.assertEqual(meta["completion_tokens"], 10)
        self.assertEqual(meta["total_tokens"], 160)
        self.assertEqual(meta["status"], "success")
        self.assertEqual(meta["error_message"], "")
        self.assertGreaterEqual(meta["latency_seconds"], 0.0)
        self.assertEqual(meta["request_id"], "resp_123")
        self.assertEqual(meta["model_version"], "gpt-test-2025")
        self.assertEqual(meta["retry_count"], 1)
        self.assertTrue(meta["started_at_utc"].endswith("Z"))
        self.assertTrue(meta["ended_at_utc"].endswith("Z"))

    @patch('ctr_labeling.llm_client.OpenAI')
    def test_get_labels_malformed_json(self, mock_openai):
        """Test that malformed JSON from the API is handled gracefully (fallback to zeros)."""
        mock_response = MagicMock()
        mock_response.choices[0].message.content = "Not a JSON string"
        # Even on failure to parse content, we might still get token usage if the API returned
        mock_response.usage.prompt_tokens = 100
        mock_response.usage.completion_tokens = 5
        mock_response.usage.total_tokens = None
        
        mock_instance = mock_openai.return_value
        mock_instance.chat.completions.create.return_value = mock_response
        
        client = LLMClient(self.cfg)
        labels, meta = client.get_labels("Report text")
        
        self.assertEqual(labels, self.empty_labels)
        self.assertEqual(meta["prompt_tokens"], 100)
        self.assertEqual(meta["completion_tokens"], 5)
        self.assertEqual(meta["total_tokens"], 105)
        self.assertEqual(meta["status"], "error")
        self.assertIn("Malformed JSON", meta["error_message"])
        self.assertGreaterEqual(meta["latency_seconds"], 0.0)
        self.assertEqual(meta["retry_count"], 1)
        self.assertTrue(meta["started_at_utc"].endswith("Z"))
        self.assertTrue(meta["ended_at_utc"].endswith("Z"))

    @patch('ctr_labeling.llm_client.OpenAI')
    def test_get_labels_api_error_fallback(self, mock_openai):
        """Test that if the API raises an unrecoverable error, we fallback to zeros."""
        mock_instance = mock_openai.return_value
        mock_instance.chat.completions.create.side_effect = Exception("API Down")
        
        client = LLMClient(self.cfg)
        labels, meta = client.get_labels("Report text")
        
        self.assertEqual(labels, self.empty_labels)
        self.assertEqual(meta["prompt_tokens"], 0)
        self.assertEqual(meta["completion_tokens"], 0)
        self.assertEqual(meta["total_tokens"], 0)
        self.assertEqual(meta["status"], "error")
        self.assertIn("API Down", meta["error_message"])
        self.assertEqual(meta["retry_count"], 1)
        self.assertTrue(meta["started_at_utc"].endswith("Z"))
        self.assertTrue(meta["ended_at_utc"].endswith("Z"))

if __name__ == '__main__':
    unittest.main()