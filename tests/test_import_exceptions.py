import pytest
from unittest.mock import patch, MagicMock


class TestImportExceptions:
    """Test suite for import exception handling in factory.py

    This test suite covers:
    1. Optional import handling for GoogleAI, Ollama, and LMStudio models
    2. Model factory behavior when model classes are None due to import failures
    3. Exception handling when creating models with invalid connection types
    4. Verification that core models are always available
    """

    @pytest.fixture(autouse=True)
    def setup_and_cleanup(self):
        """Clear cache before and after each test to ensure isolation"""
        from src.ai_factory_model.llm.factory import cache
        cache.clear()
        yield
        cache.clear()

    def test_google_ai_import_success(self):
        """Test that GoogleAI models are available when import succeeds"""
        # Test with successful imports
        with patch('src.ai_factory_model.llm.model_GoogleAIChat.GoogleAIChatModel') as mock_google_chat, \
             patch('src.ai_factory_model.llm.model_GoogleAIEmbedding.GoogleAIEmbeddingModel') as mock_google_embedding:

            # Mock the classes
            mock_google_chat.return_value = MagicMock()
            mock_google_embedding.return_value = MagicMock()

            # Import factory after mocking
            from src.ai_factory_model.llm import factory

            # Verify models are available
            assert hasattr(factory, 'GoogleAIChatModel')
            assert hasattr(factory, 'GoogleAIEmbeddingModel')

    # def test_google_ai_import_failure_handling(self):
    #     """Test that import exception handling is covered for GoogleAI models"""
    #     # This tests the exception handling code path by checking the try-except structure
    #     import ast
    #     import inspect

    #     from src.ai_factory_model.llm import factory

    #     # Get the source code of the factory module
    #     source = inspect.getsource(factory)
    #     tree = ast.parse(source)

    #     # Look for try-except blocks that handle ImportError
    #     import_error_handlers = []

    #     for node in ast.walk(tree):
    #         if isinstance(node, ast.ExceptHandler):
    #             if (node.type and isinstance(node.type, ast.Name) and node.type.id == 'ImportError'):
    #                 import_error_handlers.append(node)

    #     # Verify we have at least 3 ImportError handlers (for the 3 optional model groups)
    #     assert len(import_error_handlers) >= 3, "Should have ImportError handlers for optional models"

    def test_ollama_import_success(self):
        """Test that Ollama models are available when import succeeds"""
        with patch('src.ai_factory_model.llm.model_OllamaChat.OllamaChatModel') as mock_ollama:
            mock_ollama.return_value = MagicMock()

            from src.ai_factory_model.llm import factory

            assert hasattr(factory, 'OllamaChatModel')

    def test_lmstudio_import_success(self):
        """Test that LMStudio models are available when import succeeds"""
        with patch('src.ai_factory_model.llm.model_LMStudioChat.LMStudioChat') as mock_lmstudio:
            mock_lmstudio.return_value = MagicMock()

            from src.ai_factory_model.llm import factory

            assert hasattr(factory, 'LMStudioChat')

    def test_model_factory_handles_none_model_class(self):
        """Test that ModelFactory handles None model classes gracefully"""
        from src.ai_factory_model.llm.factory import ModelFactory

        # Test with a model class that could be None (simulating failed import)
        model_def = {
            "connection_type": "NonExistentModel",
            "model_name": "test_model",
            "api_endpoint": "https://example.com"
        }

        # This should raise an exception when model class is not found/None
        with pytest.raises(Exception):
            ModelFactory.create_model("test_alias", model_def)

    def test_model_classes_dictionary_structure(self):
        """Test that MODEL_CLASSES dictionary has expected structure"""
        from src.ai_factory_model.llm.factory import ModelFactory

        expected_keys = [
            "AzureOpenAIChat", "AzureOpenAIEmbedding",
            "OpenChatAIChat", "OpenAIEmbedding", "AzureAIChat",
            "GoogleAIChat", "GoogleAIEmbedding",
            "OllamaChat", "LMStudioChat"
        ]

        for key in expected_keys:
            assert key in ModelFactory.MODEL_CLASSES, f"Missing model class key: {key}"

    # def test_import_error_sets_models_to_none(self):
    #     """Test that when imports fail, models are set to None"""
    #     # This test verifies the behavior by examining the actual module state
    #     from src.ai_factory_model.llm import factory

    #     # Check if any optional models are None (indicating import failure occurred)
    #     optional_models = [
    #         (factory.GoogleAIChatModel, 'GoogleAIChatModel'),
    #         (factory.GoogleAIEmbeddingModel, 'GoogleAIEmbeddingModel'),
    #         (factory.OllamaChatModel, 'OllamaChatModel'),
    #         (factory.LMStudioChat, 'LMStudioChat')
    #     ]

    #     # At least one should potentially be None if imports failed
    #     # This test documents the expected behavior when imports fail
    #     for model, name in optional_models:
    #         if model is None:
    #             # If a model is None, verify it's also None in MODEL_CLASSES
    #             model_key = name.replace('Model', '').replace('Chat', 'Chat')
    #             if model_key == 'GoogleAI':
    #                 model_key = 'GoogleAIChat'  # Handle GoogleAI naming
    #             elif model_key == 'GoogleAIEmbedding':
    #                 model_key = 'GoogleAIEmbedding'
    #             elif model_key == 'Ollama':
    #                 model_key = 'OllamaChat'
    #             elif model_key == 'LMStudio':
    #                 model_key = 'LMStudioChat'

    #             # The corresponding MODEL_CLASSES entry should also be None
    #             if model_key in factory.ModelFactory.MODEL_CLASSES:
    #                 assert factory.ModelFactory.MODEL_CLASSES[model_key] is None

    @patch('src.ai_factory_model.llm.factory.load_from_file')
    def test_get_model_with_missing_connection_type(self, mock_load_from_file):
        """Test ModelFactory.get_model with missing connection_type"""
        from src.ai_factory_model.llm.factory import ModelFactory, cache

        # Clear cache to ensure fresh test
        cache.clear()

        mock_load_from_file.return_value = {
            "test_alias": {
                "connection_type": "NonExistentModel",
                "model_name": "test_model"
            }
        }

        # When model class is not found (None), it should raise an exception
        with pytest.raises(Exception) as exc_info:
            ModelFactory.get_model("test_alias")

        # The exception should contain information about the error
        assert "Error in ModelFactory.create" in str(exc_info.value)

    def test_core_models_not_none(self):
        """Test that core (required) models are never None"""
        from src.ai_factory_model.llm.factory import ModelFactory

        # These models should always be available (not optional)
        core_models = [
            "AzureOpenAIChat", "AzureOpenAIEmbedding",
            "OpenChatAIChat", "OpenAIEmbedding", "AzureAIChat"
        ]

        for model_name in core_models:
            assert ModelFactory.MODEL_CLASSES[model_name] is not None, f"Core model {model_name} should not be None"
