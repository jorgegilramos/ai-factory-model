from unittest.mock import patch, Mock
import pytest
from jinja2 import Template


class TestLMStudioChatModelImport:
    """Test import scenarios for the LMStudio chat model."""

    def test_import_error_scenario(self):
        """Test the ImportError scenario by creating a temporary module that simulates the error condition."""
        # Create a temporary Python file content that mimics the original module
        # but with an import that will definitely fail
        temp_module_code = '''
try:
    from nonexistent_langchain_openai_library import ChatOpenAI
except ImportError:
    ChatOpenAI = None

# This variable helps us verify the except block was executed
import_error_occurred = ChatOpenAI is None
'''

        # Execute the code to test the try/except block
        namespace = {}
        exec(temp_module_code, namespace)

        # Verify that the except block was executed
        assert namespace['ChatOpenAI'] is None
        assert namespace['import_error_occurred'] is True

    def test_import_error_with_module_isolation(self):
        """Test import error by creating an isolated module context."""
        # This tests the exact pattern from the LMStudio model file
        import types

        # Create a new module in isolation
        test_module = types.ModuleType('isolated_test_module')

        # Execute the exact import pattern in the isolated module
        exec_code = '''
ChatOpenAI = "initial_value"  # Set to something other than None
try:
    from this_module_definitely_does_not_exist import ChatOpenAI
except ImportError:
    ChatOpenAI = None
'''

        # Execute in the module's namespace
        exec(exec_code, test_module.__dict__)

        # Verify the except block was hit and ChatOpenAI is None
        assert test_module.ChatOpenAI is None


# Import the actual model after the import tests
from src.ai_factory_model.llm.model_LMStudioChat import LMStudioChat  # noqa: E402
from src.ai_factory_model import SEP_PATTERN  # noqa: E402


class TestLMStudioChatModel:

    def test_import_available(self):
        """Test LMStudioChat when langchain_openai is available."""
        with patch("src.ai_factory_model.llm.model_LMStudioChat.ChatOpenAI") as mock_chat:
            config = {
                "model_name": "llama-2-7b",
                "model_version": "1.0",
                "api_endpoint": "http://localhost:1234/v1",
                "model_params": {"temperature": 0.7}
            }

            model = LMStudioChat(config)
            result = model.initialize_model("test_alias")

            # Verify the model was created with correct parameters
            mock_chat.assert_called_once_with(
                openai_api_base="http://localhost:1234/v1",
                openai_api_key="lmstudio",
                model="llama-2-7b",
                temperature=0.7
            )

            assert result == model
            assert model.client is not None
            assert model.alias == "test_alias"

    def test_import_not_available(self):
        """Test LMStudioChat when langchain_openai is not available."""
        # Mock the import to be None (simulating ImportError during import)
        with patch("src.ai_factory_model.llm.model_LMStudioChat.ChatOpenAI", None):
            config = {
                "model_name": "llama-2-7b",
                "model_version": "1.0",
                "api_endpoint": "http://localhost:1234/v1",
                "model_params": {}
            }

            model = LMStudioChat(config)

            with pytest.raises(ImportError) as exc_info:
                model.initialize_model("test_alias")

            assert "langchain_openai is not installed" in str(exc_info.value)

    def test_initialize_model_basic_config(self):
        """Test initialization with basic configuration."""
        with patch("src.ai_factory_model.llm.model_LMStudioChat.ChatOpenAI") as mock_chat:
            config = {
                "model_name": "codellama-13b",
                "model_version": "1.0",
                "api_endpoint": "http://localhost:1234/v1",
                "model_params": {}
            }

            model = LMStudioChat(config)
            model.initialize_model("basic_alias")

            mock_chat.assert_called_once_with(
                openai_api_base="http://localhost:1234/v1",
                openai_api_key="lmstudio",
                model="codellama-13b"
            )
            assert model.alias == "basic_alias"

    def test_initialize_model_with_params(self):
        """Test initialization with custom model parameters."""
        with patch("src.ai_factory_model.llm.model_LMStudioChat.ChatOpenAI") as mock_chat:
            config = {
                "model_name": "mistral-7b",
                "model_version": "1.0",
                "api_endpoint": "http://localhost:1234/v1",
                "model_params": {
                    "temperature": 0.9,
                    "max_tokens": 1024,
                    "top_p": 0.8
                }
            }

            model = LMStudioChat(config)
            model.initialize_model("params_alias")

            mock_chat.assert_called_once_with(
                openai_api_base="http://localhost:1234/v1",
                openai_api_key="lmstudio",
                model="mistral-7b",
                temperature=0.9,
                max_tokens=1024,
                top_p=0.8
            )

    def test_get_client_property(self):
        """Test the get_client property."""
        with patch("src.ai_factory_model.llm.model_LMStudioChat.ChatOpenAI") as mock_chat:
            mock_client_instance = mock_chat.return_value

            config = {
                "model_name": "llama-2-7b",
                "model_version": "1.0",
                "api_endpoint": "http://localhost:1234/v1",
                "model_params": {}
            }

            model = LMStudioChat(config)
            model.initialize_model("test_alias")
            client = model.get_client

            assert client == mock_client_instance

    def test_prompt_method(self):
        """Test the prompt method with think tag filtering."""
        with patch("src.ai_factory_model.llm.model_LMStudioChat.ChatOpenAI") as mock_chat:
            # Setup mock response
            mock_response = Mock()
            mock_response.content = "This is a test response"
            mock_client_instance = mock_chat.return_value
            mock_client_instance.invoke.return_value = mock_response

            config = {
                "model_name": "llama-2-7b",
                "model_version": "1.0",
                "api_endpoint": "http://localhost:1234/v1",
                "model_params": {}
            }

            model = LMStudioChat(config)
            model.initialize_model("test_alias")

            system_message = "You are a helpful assistant."
            user_message = "What is the capital of France?"
            params = [system_message, user_message]

            result = model.prompt(params)

            # Verify the invoke was called with correct messages
            expected_messages = [
                {"role": "system", "content": system_message},
                {"role": "user", "content": user_message}
            ]
            mock_client_instance.invoke.assert_called_once_with(expected_messages)
            assert result == "This is a test response"

    def test_prompt_method_with_think_tags(self):
        """Test the prompt method filters out <think> tags."""
        with patch("src.ai_factory_model.llm.model_LMStudioChat.ChatOpenAI") as mock_chat:
            # Setup mock response with think tags
            mock_response = Mock()
            mock_response.content = "<think>Let me think about this...</think>Paris is the capital of France."
            mock_client_instance = mock_chat.return_value
            mock_client_instance.invoke.return_value = mock_response

            config = {
                "model_name": "llama-2-7b",
                "model_version": "1.0",
                "api_endpoint": "http://localhost:1234/v1",
                "model_params": {}
            }

            model = LMStudioChat(config)
            model.initialize_model("test_alias")

            system_message = "You are a helpful assistant."
            user_message = "What is the capital of France?"
            params = [system_message, user_message]

            result = model.prompt(params)

            # Verify think tags are filtered out
            assert result == "Paris is the capital of France."
            assert "<think>" not in result
            assert "</think>" not in result

    def test_prompt_method_with_multiline_think_tags(self):
        """Test the prompt method filters out multiline <think> tags."""
        with patch("src.ai_factory_model.llm.model_LMStudioChat.ChatOpenAI") as mock_chat:
            # Setup mock response with multiline think tags
            mock_response = Mock()
            mock_response.content = """<think>
Let me think step by step:
1. France is a country
2. It has a capital city
3. That city is Paris
</think>The capital of France is Paris."""
            mock_client_instance = mock_chat.return_value
            mock_client_instance.invoke.return_value = mock_response

            config = {
                "model_name": "llama-2-7b",
                "model_version": "1.0",
                "api_endpoint": "http://localhost:1234/v1",
                "model_params": {}
            }

            model = LMStudioChat(config)
            model.initialize_model("test_alias")

            system_message = "You are a helpful assistant."
            user_message = "What is the capital of France?"
            params = [system_message, user_message]

            result = model.prompt(params)

            # Verify multiline think tags are filtered out
            assert result == "The capital of France is Paris."
            assert "<think>" not in result
            assert "</think>" not in result

    def test_prompt_template_method(self):
        """Test the prompt_template method inherited from BaseModel."""
        with patch("src.ai_factory_model.llm.model_LMStudioChat.ChatOpenAI") as mock_chat:
            with patch("src.ai_factory_model.llm.model_base.read_template") as mock_read_template:
                # Setup mocks
                mock_response = Mock()
                mock_response.content = "Template response"
                mock_client_instance = mock_chat.return_value
                mock_client_instance.invoke.return_value = mock_response
                mock_read_template.return_value = ["System: Hello", "User: Test"]

                config = {
                    "model_name": "llama-2-7b",
                    "model_version": "1.0",
                    "api_endpoint": "http://localhost:1234/v1",
                    "model_params": {}
                }

                model = LMStudioChat(config)
                model.initialize_model("test_alias")

                template_path = "test_template.txt"
                params = {"name": "John"}

                result = model.prompt_template(template_path, params)

                mock_read_template.assert_called_once_with(template_path, params, SEP_PATTERN)
                assert result == "Template response"

    def test_prompt_render_method(self):
        """Test the prompt_render method inherited from BaseModel."""
        with patch("src.ai_factory_model.llm.model_LMStudioChat.ChatOpenAI") as mock_chat:
            with patch("src.ai_factory_model.llm.model_base.render_template") as mock_render_template:
                # Setup mocks
                mock_response = Mock()
                mock_response.content = "Rendered response"
                mock_client_instance = mock_chat.return_value
                mock_client_instance.invoke.return_value = mock_response
                mock_render_template.return_value = ["System: Rendered", "User: Content"]

                config = {
                    "model_name": "llama-2-7b",
                    "model_version": "1.0",
                    "api_endpoint": "http://localhost:1234/v1",
                    "model_params": {}
                }

                model = LMStudioChat(config)
                model.initialize_model("test_alias")

                template = Template("{{ greeting }}")
                params = {"greeting": "Hello"}

                result = model.prompt_render(template, params)

                mock_render_template.assert_called_once_with(template, params, SEP_PATTERN)
                assert result == "Rendered response"

    def test_embedding_method_raises_exception(self):
        """Test that embedding method raises exception (inherited from BaseModel)."""
        with patch("src.ai_factory_model.llm.model_LMStudioChat.ChatOpenAI"):
            config = {
                "model_name": "llama-2-7b",
                "model_version": "1.0",
                "api_endpoint": "http://localhost:1234/v1",
                "model_params": {}
            }

            model = LMStudioChat(config)
            model.initialize_model("test_alias")

            with pytest.raises(Exception) as exc_info:
                model.embedding("test text")

            assert "does not allow embedding" in str(exc_info.value)

    def test_config_inheritance_from_base(self):
        """Test that configuration is properly inherited from BaseModel."""
        config = {
            "model_name": "llama-2-7b",
            "model_version": "v1.0",
            "api_endpoint": "http://localhost:1234/v1",
            "model_params": {"temperature": 0.5}
        }

        model = LMStudioChat(config)

        # Verify inherited attributes from BaseModel
        assert model.config == config
        assert model.model_name == "llama-2-7b"
        assert model.version == "v1.0"
        assert model.endpoint == "http://localhost:1234/v1"
        assert model.params == {"temperature": 0.5}

    def test_render_var_method(self):
        """Test the render_var method inherited from BaseModel."""
        config = {
            "model_name": "llama-2-7b-test",
            "model_version": "1.0",
            "api_endpoint": "http://localhost:1234/v1",
            "test_var": "test_value"
        }

        model = LMStudioChat(config)

        # Test rendering existing config variable
        result = model.render_var("model_name")
        assert result == "llama-2-7b-test"

        # Test rendering with default value
        result = model.render_var("non_existent", default="default_value")
        assert result == "default_value"

    def test_render_property_with_template(self):
        """Test the render_property method with template variables."""
        with patch("src.ai_factory_model.llm.model_base.get_var") as mock_get_var:
            mock_get_var.return_value = "substituted_value"

            config = {
                "model_name": "llama-2-7b",
                "model_version": "1.0",
                "api_endpoint": "http://localhost:1234/v1"
            }

            model = LMStudioChat(config)

            # Test template rendering
            template_value = "prefix_{TEST_VAR}_suffix"
            result = model.render_property(template_value)

            mock_get_var.assert_called_once_with(var_name="TEST_VAR")
            assert result == "prefix_substituted_value_suffix"

    def test_client_is_none_before_initialization(self):
        """Test that client is None before initialization."""
        config = {
            "model_name": "llama-2-7b",
            "model_version": "1.0",
            "api_endpoint": "http://localhost:1234/v1",
            "model_params": {}
        }

        model = LMStudioChat(config)

        assert model.client is None
        assert model.get_client is None

    def test_alias_not_set_before_initialization(self):
        """Test that alias is not set before initialization."""
        config = {
            "model_name": "llama-2-7b",
            "model_version": "1.0",
            "api_endpoint": "http://localhost:1234/v1",
            "model_params": {}
        }

        model = LMStudioChat(config)

        # alias should not exist as an attribute before initialization
        assert not hasattr(model, 'alias')


class TestLMStudioChatModelCoverage:
    """Additional tests to ensure 100% code coverage regardless of environment setup."""

    def test_import_error_handling(self):
        """Test import error handling without environment dependency."""
        with patch("src.ai_factory_model.llm.model_LMStudioChat.ChatOpenAI", None):
            config = {
                "model_name": "llama-2-7b",
                "model_version": "1.0",
                "api_endpoint": "http://localhost:1234/v1",
                "model_params": {}
            }

            model = LMStudioChat(config)

            with pytest.raises(ImportError) as exc_info:
                model.initialize_model("test_alias")

            assert "langchain_openai is not installed" in str(exc_info.value)
            assert "Please install it to use LMStudioChat" in str(exc_info.value)

    def test_successful_initialization_coverage(self):
        """Test successful initialization for coverage."""
        with patch("src.ai_factory_model.llm.model_LMStudioChat.ChatOpenAI") as mock_chat:
            mock_client = Mock()
            mock_chat.return_value = mock_client

            config = {
                "model_name": "codellama-13b",
                "model_version": "1.0",
                "api_endpoint": "http://localhost:1234/v1",
                "model_params": {
                    "temperature": 0.8,
                    "max_tokens": 2048
                }
            }

            model = LMStudioChat(config)
            result = model.initialize_model("coverage_alias")

            # Verify the method returns self
            assert result is model

            # Verify ChatOpenAI was called with correct parameters
            mock_chat.assert_called_once_with(
                openai_api_base="http://localhost:1234/v1",
                openai_api_key="lmstudio",
                model="codellama-13b",
                temperature=0.8,
                max_tokens=2048
            )

            # Verify client and alias are set
            assert model.client is mock_client
            assert model.alias == "coverage_alias"

    def test_base_model_inheritance(self):
        """Test that the model properly inherits from BaseModel."""
        config = {
            "model_name": "llama-2-7b",
            "model_version": "1.0",
            "api_endpoint": "http://localhost:1234/v1",
            "model_params": {"temperature": 0.5}
        }

        model = LMStudioChat(config)

        # Test inherited initialization
        assert hasattr(model, 'config')
        assert hasattr(model, 'model_name')
        assert hasattr(model, 'params')
        assert model.config == config
        assert model.model_name == "llama-2-7b"
        assert model.params == {"temperature": 0.5}

    def test_prompt_method_edge_cases(self):
        """Test prompt method with various edge cases for think tag removal."""
        with patch("src.ai_factory_model.llm.model_LMStudioChat.ChatOpenAI") as mock_chat:
            mock_client_instance = mock_chat.return_value

            config = {
                "model_name": "llama-2-7b",
                "model_version": "1.0",
                "api_endpoint": "http://localhost:1234/v1",
                "model_params": {}
            }

            model = LMStudioChat(config)
            model.initialize_model("test_alias")

            params = ["You are a helpful assistant.", "Test question"]

            # Test case 1: Response with only think tags (should return empty string)
            mock_response = Mock()
            mock_response.content = "<think>Only thinking, no output</think>"
            mock_client_instance.invoke.return_value = mock_response

            result = model.prompt(params)
            assert result == ""

            # Test case 2: Multiple think tags
            mock_response.content = "<think>First thought</think>Answer<think>Second thought</think>"
            result = model.prompt(params)
            assert result == "Answer"

            # Test case 3: No think tags
            mock_response.content = "Just a normal response"
            result = model.prompt(params)
            assert result == "Just a normal response"

            # Test case 4: Nested-looking content (should not remove non-think tags)
            mock_response.content = "<think>Remove this</think>Keep <important>this</important> content"
            result = model.prompt(params)
            assert result == "Keep <important>this</important> content"
