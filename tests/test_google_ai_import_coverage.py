"""
Comprehensive test for GoogleAI chat model import error coverage.
This test specifically targets the try/except ImportError block in the original module.
"""

import sys
# import importlib
# from unittest.mock import patch
import pytest


def test_google_ai_import_error_with_reload():
    """
    Test that covers the ImportError exception handling in the GoogleAI chat model
    by simulating a missing langchain_google_genai dependency.
    """

    # Store the original module if it exists
    original_module = sys.modules.get('src.ai_factory_model.llm.model_GoogleAIChat')

    try:
        # Remove the module from cache if it exists
        if 'src.ai_factory_model.llm.model_GoogleAIChat' in sys.modules:
            del sys.modules['src.ai_factory_model.llm.model_GoogleAIChat']

        # Also remove any langchain modules that might be cached
        modules_to_remove = [name for name in sys.modules.keys() if 'langchain_google_genai' in name]
        for module_name in modules_to_remove:
            del sys.modules[module_name]

        # Mock the langchain_google_genai import to raise ImportError
        def mock_import(name, *args, **kwargs):
            if 'langchain_google_genai' in name:
                raise ImportError(f"No module named '{name}'")
            # For all other imports, use the original import mechanism
            return original_import(name, *args, **kwargs)

        # Store original import
        original_import = __builtins__['__import__']

        try:
            # Apply the mock
            __builtins__['__import__'] = mock_import

            # Now import the module, which should trigger the except ImportError block
            import src.ai_factory_model.llm.model_GoogleAIChat as test_module

            # Verify that ChatGoogleGenerativeAI is None (indicating except block was hit)
            assert test_module.ChatGoogleGenerativeAI is None

            # Also verify the module can be instantiated and raises appropriate error
            config = {
                "model_name": "gemini-pro",
                "model_version": "1.0",
                "api_key": "test_key",
                "api_auth": "api_key",
                "model_params": {}
            }

            model = test_module.GoogleAIChatModel(config)

            # This should raise ImportError when initialize_model is called
            with pytest.raises(ImportError) as exc_info:
                model.initialize_model("test_alias")

            assert "langchain_google_genai is not installed" in str(exc_info.value)

        finally:
            # Restore original import
            __builtins__['__import__'] = original_import

    finally:
        # Clean up: remove the test module from cache
        if 'src.ai_factory_model.llm.model_GoogleAIChat' in sys.modules:
            del sys.modules['src.ai_factory_model.llm.model_GoogleAIChat']

        # Restore original module if it existed
        if original_module is not None:
            sys.modules['src.ai_factory_model.llm.model_GoogleAIChat'] = original_module


if __name__ == "__main__":
    test_google_ai_import_error_with_reload()
    print("Import error coverage test with reload passed!")
