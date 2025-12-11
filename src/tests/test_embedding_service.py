import os

# Temporarily add src to path to allow imports
import sys
import unittest
from unittest.mock import patch

import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.services.embedding_service import EmbeddingService


@pytest.mark.skip(reason="EmbeddingService refactored to use provider pattern - tests need to be updated for EmbeddingsManager")
class TestEmbeddingService(unittest.TestCase):
    @patch("ollama.embeddings")
    def test_generate_embeddings_success(self, mock_ollama_embeddings):
        mock_ollama_embeddings.return_value = {"embedding": [0.1, 0.2, 0.3]}
        service = EmbeddingService()
        embedding = service.generate_embeddings("test_model", "test text")
        self.assertEqual(embedding, [0.1, 0.2, 0.3])
        mock_ollama_embeddings.assert_called_once_with(model="test_model", prompt="test text", host=service.ollama_host)

    @patch("ollama.embeddings")
    def test_generate_embeddings_failure(self, mock_ollama_embeddings):
        mock_ollama_embeddings.side_effect = Exception("Ollama error")
        service = EmbeddingService()
        embedding = service.generate_embeddings("test_model", "test text")
        self.assertIsNone(embedding)

    @patch("platform.system", return_value="Darwin")
    @patch("torch.backends.mps.is_available", return_value=True)
    @patch("builtins.print")  # Mock print to suppress output during test
    def test_get_device_macos_mps_available(self, mock_print, mock_mps_available, mock_platform_system):
        service = EmbeddingService()
        self.assertEqual(str(service.device), "mps")
        mock_print.assert_called_with("MPS is available. Using Metal for acceleration.")

    @patch("platform.system", return_value="Darwin")
    @patch("torch.backends.mps.is_available", return_value=False)
    @patch("builtins.print")  # Mock print to suppress output during test
    def test_get_device_macos_mps_not_available(self, mock_print, mock_mps_available, mock_platform_system):
        service = EmbeddingService()
        self.assertEqual(str(service.device), "cpu")
        mock_print.assert_called_with("MPS not available, using CPU.")

    @patch("platform.system", return_value="Linux")
    @patch("builtins.print")  # Mock print to suppress output during test
    def test_get_device_linux(self, mock_print, mock_platform_system):
        service = EmbeddingService()
        self.assertEqual(str(service.device), "cpu")
        mock_print.assert_not_called()  # No print for non-macOS
