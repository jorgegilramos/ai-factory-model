from unittest.mock import patch, Mock
import pytest
from jinja2 import Template

from src.ai_factory_model.llm.model_AzureOpenAIChat import AzureOpenAIChatModel
from src.ai_factory_model import SEP_PATTERN


class TestAzureOpenAIChatModelImport:
    """Test import scenarios for the AzureOpenAI chat model."""

    def test_import_success_scenario(self):
        """Test that AzureOpenAI imports successfully (no try/except like GoogleAI)."""
        # Unlike GoogleAI, AzureOpenAI doesn't have conditional imports
        # This test verifies the import is always available
        from src.ai_factory_model.llm.model_AzureOpenAIChat import AzureChatOpenAI

        # AzureChatOpenAI should be directly imported (no None fallback)
        assert AzureChatOpenAI is not None

    def test_import_verification(self):
        """Test that all necessary imports are available for AzureOpenAI model."""
        # Verify all required imports exist
        from src.ai_factory_model.llm.model_AzureOpenAIChat import (
            AzureOpenAIChatModel,
            AzureChatOpenAI,
            BaseModel,
            VALUE_SERVICE_PRINCIPAL,
            VALUE_API_KEY,
            AzureAuthClient
        )

        assert AzureOpenAIChatModel is not None
        assert AzureChatOpenAI is not None
        assert BaseModel is not None
        assert VALUE_SERVICE_PRINCIPAL == "service_principal"
        assert VALUE_API_KEY == "api_key"
        assert AzureAuthClient is not None


class TestAzureOpenAIChatModel:

    def test_initialize_model_service_principal(self):
        """Test AzureOpenAIChatModel with service principal authentication."""
        with patch("src.ai_factory_model.llm.model_AzureOpenAIChat.AzureChatOpenAI") as mock_azure_chat:
            with patch("src.ai_factory_model.llm.model_AzureOpenAIChat.AzureAuthClient") as mock_auth_client:
                mock_auth_client_instance = mock_auth_client.return_value
                mock_auth_client_instance.get_token_provider.return_value = "mock_token_provider"

                config = {
                    "model_name": "gpt-4",
                    "model_version": "2023-12-01-preview",
                    "api_auth": "service_principal",
                    "api_endpoint": "https://test-openai.openai.azure.com/",
                    "model_params": {"temperature": 0.7}
                }

                model = AzureOpenAIChatModel(config)
                result = model.initialize_model("test_alias")

                # Verify the model was created with correct parameters for service principal
                mock_azure_chat.assert_called_once_with(
                    azure_endpoint="https://test-openai.openai.azure.com/",
                    azure_ad_token_provider="mock_token_provider",
                    azure_deployment="gpt-4",
                    api_version="2023-12-01-preview",
                    temperature=0.7
                )

                assert result == model
                assert model.client is not None
                assert model.alias == "test_alias"
                assert model.auth_client is not None

    def test_initialize_model_api_key(self):
        """Test AzureOpenAIChatModel with API key authentication."""

        with patch("src.ai_factory_model.llm.model_AzureOpenAIChat.AzureChatOpenAI") as mock_azure_chat:
            config = {
                "model_name": "gpt-35-turbo",
                "model_version": "2023-05-15",
                "api_auth": "api_key",
                "api_key": "test_api_key",
                "api_endpoint": "https://test-openai.openai.azure.com/",
                "model_params": {}
            }

            model = AzureOpenAIChatModel(config)
            result = model.initialize_model("api_key_alias")

            # Verify the model was created with correct parameters for API key
            mock_azure_chat.assert_called_once_with(
                azure_endpoint="https://test-openai.openai.azure.com/",
                api_key="test_api_key",
                azure_deployment="gpt-35-turbo",
                api_version="2023-05-15"
            )

            assert result == model
            assert model.client is not None
            assert model.alias == "api_key_alias"

    def test_initialize_model_invalid_auth(self):
        """Test AzureOpenAIChatModel with invalid authentication type."""

        config = {
            "model_name": "gpt-4",
            "model_version": "2023-12-01-preview",
            "api_auth": "invalid_auth",
            "api_endpoint": "https://test-openai.openai.azure.com/",
            "model_params": {}
        }

        model = AzureOpenAIChatModel(config)

        with pytest.raises(ValueError) as exc_info:
            model.initialize_model("test_alias")

        assert "Authorization should be \"service_principal\" or \"api_key\"" in str(exc_info.value)

    def test_initialize_model_basic_config(self):
        """Test initialization with basic configuration."""

        with patch("src.ai_factory_model.llm.model_AzureOpenAIChat.AzureChatOpenAI") as mock_azure_chat:
            config = {
                "model_name": "gpt-4-turbo",
                "model_version": "2024-02-01",
                "api_auth": "api_key",
                "api_key": "test_api_key_basic",
                "api_endpoint": "https://basic-test.openai.azure.com/",
                "model_params": {}
            }

            model = AzureOpenAIChatModel(config)
            model.initialize_model("basic_alias")

            mock_azure_chat.assert_called_once_with(
                azure_endpoint="https://basic-test.openai.azure.com/",
                api_key="test_api_key_basic",
                azure_deployment="gpt-4-turbo",
                api_version="2024-02-01"
            )
            assert model.alias == "basic_alias"

    def test_initialize_model_with_params(self):
        """Test initialization with custom model parameters."""
        with patch("src.ai_factory_model.llm.model_AzureOpenAIChat.AzureChatOpenAI") as mock_azure_chat:
            config = {
                "model_name": "gpt-4",
                "model_version": "2023-12-01-preview",
                "api_auth": "api_key",
                "api_key": "test_api_key_params",
                "api_endpoint": "https://params-test.openai.azure.com/",
                "model_params": {
                    "temperature": 0.9,
                    "max_tokens": 1024,
                    "top_p": 0.8,
                    "frequency_penalty": 0.1
                }
            }

            model = AzureOpenAIChatModel(config)
            model.initialize_model("params_alias")

            mock_azure_chat.assert_called_once_with(
                azure_endpoint="https://params-test.openai.azure.com/",
                api_key="test_api_key_params",
                azure_deployment="gpt-4",
                api_version="2023-12-01-preview",
                temperature=0.9,
                max_tokens=1024,
                top_p=0.8,
                frequency_penalty=0.1
            )

    def test_get_client_property(self):
        """Test the get_client property."""
        with patch("src.ai_factory_model.llm.model_AzureOpenAIChat.AzureChatOpenAI") as mock_azure_chat:
            mock_client_instance = mock_azure_chat.return_value

            config = {
                "model_name": "gpt-35-turbo",
                "model_version": "2023-05-15",
                "api_auth": "api_key",
                "api_key": "test_api_key",
                "api_endpoint": "https://test-openai.openai.azure.com/",
                "model_params": {}
            }

            model = AzureOpenAIChatModel(config)
            model.initialize_model("test_alias")
            client = model.get_client

            assert client == mock_client_instance

    def test_prompt_method(self):
        """Test the prompt method inherited from BaseModel."""
        with patch("src.ai_factory_model.llm.model_AzureOpenAIChat.AzureChatOpenAI") as mock_azure_chat:
            # Setup mock response
            mock_response = Mock()
            mock_response.content = "This is an Azure OpenAI response"
            mock_client_instance = mock_azure_chat.return_value
            mock_client_instance.invoke.return_value = mock_response

            config = {
                "model_name": "gpt-4",
                "model_version": "2023-12-01-preview",
                "api_auth": "api_key",
                "api_key": "test_api_key",
                "api_endpoint": "https://test-openai.openai.azure.com/",
                "model_params": {}
            }

            model = AzureOpenAIChatModel(config)
            model.initialize_model("test_alias")

            system_message = "You are a helpful Azure OpenAI assistant."
            user_message = "What is the capital of France?"
            params = [system_message, user_message]

            result = model.prompt(params)

            # Verify the invoke was called with correct messages
            expected_messages = [
                {"role": "system", "content": system_message},
                {"role": "user", "content": user_message}
            ]
            mock_client_instance.invoke.assert_called_once_with(expected_messages)
            assert result == "This is an Azure OpenAI response"

    def test_prompt_template_method(self):
        """Test the prompt_template method inherited from BaseModel."""
        with patch("src.ai_factory_model.llm.model_AzureOpenAIChat.AzureChatOpenAI") as mock_azure_chat:
            with patch("src.ai_factory_model.llm.model_base.read_template") as mock_read_template:
                # Setup mocks
                mock_response = Mock()
                mock_response.content = "Azure template response"
                mock_client_instance = mock_azure_chat.return_value
                mock_client_instance.invoke.return_value = mock_response
                mock_read_template.return_value = ["System: Azure Hello", "User: Azure Test"]

                config = {
                    "model_name": "gpt-35-turbo",
                    "model_version": "2023-05-15",
                    "api_auth": "api_key",
                    "api_key": "test_api_key",
                    "api_endpoint": "https://test-openai.openai.azure.com/",
                    "model_params": {}
                }

                model = AzureOpenAIChatModel(config)
                model.initialize_model("test_alias")

                template_path = "azure_test_template.txt"
                params = {"name": "Azure John"}

                result = model.prompt_template(template_path, params)

                mock_read_template.assert_called_once_with(template_path, params, SEP_PATTERN)
                assert result == "Azure template response"

    def test_prompt_render_method(self):
        """Test the prompt_render method inherited from BaseModel."""
        with patch("src.ai_factory_model.llm.model_AzureOpenAIChat.AzureChatOpenAI") as mock_azure_chat:
            with patch("src.ai_factory_model.llm.model_base.render_template") as mock_render_template:
                # Setup mocks
                mock_response = Mock()
                mock_response.content = "Azure rendered response"
                mock_client_instance = mock_azure_chat.return_value
                mock_client_instance.invoke.return_value = mock_response
                mock_render_template.return_value = ["System: Azure Rendered", "User: Azure Content"]

                config = {
                    "model_name": "gpt-4",
                    "model_version": "2023-12-01-preview",
                    "api_auth": "api_key",
                    "api_key": "test_api_key",
                    "api_endpoint": "https://test-openai.openai.azure.com/",
                    "model_params": {}
                }

                model = AzureOpenAIChatModel(config)
                model.initialize_model("test_alias")

                template = Template("{{ azure_greeting }}")
                params = {"azure_greeting": "Hello from Azure"}

                result = model.prompt_render(template, params)

                mock_render_template.assert_called_once_with(template, params, SEP_PATTERN)
                assert result == "Azure rendered response"

    def test_embedding_method_raises_exception(self):
        """Test that embedding method raises exception (inherited from BaseModel)."""
        with patch("src.ai_factory_model.llm.model_AzureOpenAIChat.AzureChatOpenAI"):
            config = {
                "model_name": "gpt-4",
                "model_version": "2023-12-01-preview",
                "api_auth": "api_key",
                "api_key": "test_api_key",
                "api_endpoint": "https://test-openai.openai.azure.com/",
                "model_params": {}
            }

            model = AzureOpenAIChatModel(config)
            model.initialize_model("test_alias")

            with pytest.raises(Exception) as exc_info:
                model.embedding("test text")

            assert "does not allow embedding" in str(exc_info.value)

    def test_config_inheritance_from_base(self):
        """Test that configuration is properly inherited from BaseModel."""
        config = {
            "model_name": "gpt-4",
            "model_version": "2023-12-01-preview",
            "api_key": "test_key",
            "api_auth": "api_key",
            "api_endpoint": "https://test-openai.openai.azure.com/",
            "model_params": {"temperature": 0.5}
        }

        model = AzureOpenAIChatModel(config)

        # Verify inherited attributes from BaseModel
        assert model.config == config
        assert model.model_name == "gpt-4"
        assert model.version == "2023-12-01-preview"
        assert model.api_key == "test_key"
        assert model.api_auth == "api_key"
        assert model.endpoint == "https://test-openai.openai.azure.com/"
        assert model.params == {"temperature": 0.5}

    def test_render_var_method(self):
        """Test the render_var method inherited from BaseModel."""
        config = {
            "model_name": "gpt-4-azure-test",
            "model_version": "2023-12-01-preview",
            "api_endpoint": "https://test-openai.openai.azure.com/",
            "test_var": "azure_test_value"
        }

        model = AzureOpenAIChatModel(config)

        # Test rendering existing config variable
        result = model.render_var("model_name")
        assert result == "gpt-4-azure-test"

        # Test rendering with default value
        result = model.render_var("non_existent", default="azure_default_value")
        assert result == "azure_default_value"

    def test_render_property_with_template(self):
        """Test the render_property method with template variables."""
        with patch("src.ai_factory_model.llm.model_base.get_var") as mock_get_var:
            mock_get_var.return_value = "azure_substituted_value"

            config = {
                "model_name": "gpt-4",
                "model_version": "2023-12-01-preview",
                "api_endpoint": "https://test-openai.openai.azure.com/",
                "api_key": "test_key"
            }

            model = AzureOpenAIChatModel(config)

            # Test template rendering
            template_value = "azure_prefix_{AZURE_TEST_VAR}_suffix"
            result = model.render_property(template_value)

            mock_get_var.assert_called_once_with(var_name="AZURE_TEST_VAR")
            assert result == "azure_prefix_azure_substituted_value_suffix"

    def test_client_is_none_before_initialization(self):
        """Test that client is None before initialization."""
        config = {
            "model_name": "gpt-4",
            "model_version": "2023-12-01-preview",
            "api_key": "test_key",
            "api_auth": "api_key",
            "api_endpoint": "https://test-openai.openai.azure.com/",
            "model_params": {}
        }

        model = AzureOpenAIChatModel(config)

        assert model.client is None
        assert model.get_client is None

    def test_alias_not_set_before_initialization(self):
        """Test that alias is not set before initialization."""
        config = {
            "model_name": "gpt-4",
            "model_version": "2023-12-01-preview",
            "api_key": "test_key",
            "api_auth": "api_key",
            "api_endpoint": "https://test-openai.openai.azure.com/",
            "model_params": {}
        }

        model = AzureOpenAIChatModel(config)

        # alias should not exist as an attribute before initialization
        assert not hasattr(model, 'alias')


class TestAzureOpenAIChatModelCoverage:
    """Additional tests to ensure 100% code coverage regardless of environment setup."""

    def test_service_principal_auth_coverage(self):
        """Test service principal authentication for coverage."""
        with patch("src.ai_factory_model.llm.model_AzureOpenAIChat.AzureChatOpenAI") as mock_azure_chat:
            with patch("src.ai_factory_model.llm.model_AzureOpenAIChat.AzureAuthClient") as mock_auth_client:
                mock_azure_client = Mock()
                mock_azure_chat.return_value = mock_azure_client
                mock_auth_client_instance = mock_auth_client.return_value
                mock_auth_client_instance.get_token_provider.return_value = "test_token_provider"

                config = {
                    "model_name": "gpt-4-coverage",
                    "model_version": "2023-12-01-preview",
                    "api_auth": "service_principal",
                    "api_endpoint": "https://coverage-test.openai.azure.com/",
                    "model_params": {
                        "temperature": 0.8,
                        "max_tokens": 2048
                    }
                }

                model = AzureOpenAIChatModel(config)
                result = model.initialize_model("coverage_alias")

                # Verify the method returns self
                assert result is model

                # Verify AzureChatOpenAI was called with correct parameters for service principal
                mock_azure_chat.assert_called_once_with(
                    azure_endpoint="https://coverage-test.openai.azure.com/",
                    azure_ad_token_provider="test_token_provider",
                    azure_deployment="gpt-4-coverage",
                    api_version="2023-12-01-preview",
                    temperature=0.8,
                    max_tokens=2048
                )

                # Verify client, alias, and auth_client are set
                assert model.client is mock_azure_client
                assert model.alias == "coverage_alias"
                assert model.auth_client is mock_auth_client_instance

    def test_api_key_auth_coverage(self):
        """Test API key authentication for coverage."""
        with patch("src.ai_factory_model.llm.model_AzureOpenAIChat.AzureChatOpenAI") as mock_azure_chat:
            mock_azure_client = Mock()
            mock_azure_chat.return_value = mock_azure_client

            config = {
                "model_name": "gpt-35-turbo-coverage",
                "model_version": "2023-05-15",
                "api_key": "test_coverage_api_key",
                "api_auth": "api_key",
                "api_endpoint": "https://coverage-test.openai.azure.com/",
                "model_params": {
                    "temperature": 0.3,
                    "top_p": 0.9
                }
            }

            model = AzureOpenAIChatModel(config)
            result = model.initialize_model("api_coverage_alias")

            # Verify the method returns self
            assert result is model

            # Verify AzureChatOpenAI was called with correct parameters for API key
            mock_azure_chat.assert_called_once_with(
                azure_endpoint="https://coverage-test.openai.azure.com/",
                api_key="test_coverage_api_key",
                azure_deployment="gpt-35-turbo-coverage",
                api_version="2023-05-15",
                temperature=0.3,
                top_p=0.9
            )

            # Verify client and alias are set
            assert model.client is mock_azure_client
            assert model.alias == "api_coverage_alias"

    def test_invalid_auth_error_coverage(self):
        """Test ValueError for invalid authentication type."""
        config = {
            "model_name": "gpt-4",
            "model_version": "2023-12-01-preview",
            "api_auth": "invalid_auth_type",
            "api_endpoint": "https://test-openai.openai.azure.com/",
            "model_params": {}
        }

        model = AzureOpenAIChatModel(config)

        with pytest.raises(ValueError) as exc_info:
            model.initialize_model("error_test_alias")

        assert "Authorization should be \"service_principal\" or \"api_key\"" in str(exc_info.value)

    def test_base_model_inheritance(self):
        """Test that the model properly inherits from BaseModel."""
        config = {
            "model_name": "gpt-4-inheritance",
            "model_version": "2023-12-01-preview",
            "api_key": "test_key",
            "api_auth": "api_key",
            "api_endpoint": "https://test-openai.openai.azure.com/",
            "model_params": {"temperature": 0.5}
        }

        model = AzureOpenAIChatModel(config)

        # Test inherited initialization
        assert hasattr(model, 'config')
        assert hasattr(model, 'model_name')
        assert hasattr(model, 'params')
        assert hasattr(model, 'endpoint')
        assert model.config == config
        assert model.model_name == "gpt-4-inheritance"
        assert model.version == "2023-12-01-preview"
        assert model.endpoint == "https://test-openai.openai.azure.com/"
        assert model.params == {"temperature": 0.5}

    def test_auth_client_none_before_initialization(self):
        """Test that auth_client is None before initialization with service principal."""
        config = {
            "model_name": "gpt-4",
            "model_version": "2023-12-01-preview",
            "api_auth": "service_principal",
            "api_endpoint": "https://test-openai.openai.azure.com/",
            "model_params": {}
        }

        model = AzureOpenAIChatModel(config)

        # auth_client should be None before initialization
        assert model.auth_client is None
