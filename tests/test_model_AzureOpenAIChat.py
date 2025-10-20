from unittest.mock import patch
from src.ai_factory_model.llm import AzureOpenAIChatModel
from src.ai_factory_model.llm.model_AzureOpenAIChat import AzureChatOpenAI

from src.ai_factory_model import (
    ModelFactory,
    SEP_PATTERN
)
from jinja2 import Template


def test_initialize_model_service_principal(env_testing):
    if env_testing:
        with patch("ai_factory_model.llm.model_AzureOpenAIChat.AzureAuthClient",
                   autospec=True) as mock_azure_auth_client:

            mock_auth_client_instance = mock_azure_auth_client.return_value
            mock_auth_client_instance.get_token.return_value = "mock_token"

            config = {
                "model_name": "test_model",
                "model_version": "1.0",
                "api_auth": "service_principal",
                "api_endpoint": "https://example.com",
                "model_params": {}
            }

            model = AzureOpenAIChatModel(config)
            model.initialize_model("test_alias")

            assert model.client.azure_endpoint == "https://example.com"
            assert model.client.azure_ad_token_provider is not None
            assert model.client.deployment_name == "test_model"
            assert model.client.openai_api_version == "1.0"
            assert isinstance(model.client, AzureChatOpenAI)
    else:
        assert True  # pragma: no cover


def test_initialize_model_api_key(env_testing):
    if env_testing:

        config = {
            "model_name": "test_model",
            "model_version": "1.0",
            "api_auth": "api_key",
            "api_key": "mock_api_key",
            "api_endpoint": "https://example.com",
            "model_params": {}
        }

        model = AzureOpenAIChatModel(config)
        model.initialize_model("test_alias")

        assert model.client.azure_endpoint == "https://example.com"
        assert model.client.openai_api_key is not None
        assert model.client.deployment_name == "test_model"
        assert model.client.openai_api_version == "1.0"
        assert isinstance(model.client, AzureChatOpenAI)
    else:
        assert True  # pragma: no cover


def test_initialize_model_invalid_auth(env_testing):
    if env_testing:
        config = {
            "model_name": "test_model",
            "model_version": "1.0",
            "api_auth": "invalid_auth",
            "api_endpoint": "https://example.com",
            "model_params": {}
        }

        model = AzureOpenAIChatModel(config)
        try:
            model.initialize_model("test_alias")
        except ValueError as e:
            assert str(e) == "Authorization should be \"service_principal\" or \"api_key\""
    else:
        assert True  # pragma: no cover


def test_create_model(env_testing):
    if env_testing:
        with patch("ai_factory_model.llm.model_AzureOpenAIChat.AzureAuthClient",
                   autospec=True) as mock_azure_auth_client:

            mock_auth_client_instance = mock_azure_auth_client.return_value
            mock_auth_client_instance.get_token.return_value = "mock_token"

            config = {
                "model_name": "test_model",
                "model_version": "1.0",
                "api_auth": "service_principal",
                "api_endpoint": "https://example.com",
                "model_params": {}
            }

            model = AzureOpenAIChatModel(config)
            model.initialize_model("test_alias")

            assert model.client.azure_endpoint == "https://example.com"
            assert model.client.azure_ad_token_provider is not None
            assert model.client.deployment_name == "test_model"
            assert model.client.openai_api_version == "1.0"
            assert isinstance(model.client, AzureChatOpenAI)
    else:
        assert True  # pragma: no cover


def test_read_model(env_testing):
    if env_testing:
        with patch("ai_factory_model.llm.model_AzureOpenAIChat.AzureChatOpenAI",
                   autospec=True) as mock_azure_chat_openai:

            mock_client_instance = mock_azure_chat_openai.return_value

            config = {
                "model_name": "test_model",
                "model_version": "1.0",
                "api_auth": "service_principal",
                "api_endpoint": "https://example.com",
                "model_params": {}
            }

            model = AzureOpenAIChatModel(config)
            model.client = mock_client_instance
            client = model.get_client

            assert client == mock_client_instance
    else:
        assert True  # pragma: no cover


def test_update_model(env_testing):
    if env_testing:

        config = {
            "model_name": "test_model",
            "model_version": "1.0",
            "api_auth": "service_principal",
            "api_endpoint": "https://example.com",
            "model_params": {}
        }

        model = AzureOpenAIChatModel(config)
        model.config["model_name"] = "updated_model"
        model.model_name = model.render_var("model_name")

        assert model.model_name == "updated_model"
    else:
        assert True  # pragma: no cover


def test_delete_model(env_testing):
    if env_testing:

        config = {
            "model_name": "test_model",
            "model_version": "1.0",
            "api_auth": "service_principal",
            "api_endpoint": "https://example.com",
            "model_params": {}
        }

        model = AzureOpenAIChatModel(config)
        model.client = None

        assert model.client is None
    else:
        assert True  # pragma: no cover


def test_prompt_render(env_testing):
    if env_testing:

        model = ModelFactory.get_model("azai_gtp4o")
        params = {"system": "Eres un guía turístico", "user": "¿Qué visitar en Mérida de Extremadura?"}

        template_content = (
            f"{{{{ system }}}}"
            f"{SEP_PATTERN}"
            f"{{{{ user }}}}"
        )
        prompt_template = Template(template_content)

        response = model.prompt_render(
            template=prompt_template,
            params=params,
            sep_pattern=SEP_PATTERN
        )

        assert response is not None
        assert isinstance(response, str)
    else:
        assert True  # pragma: no cover
