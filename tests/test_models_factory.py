from unittest.mock import patch
from src.ai_factory_model.llm.factory import ModelFactory, cache
from src.ai_factory_model.llm.model_AzureOpenAIChat import AzureOpenAIChatModel


def test_get_model_from_cache(env_testing):
    if env_testing:
        with patch("ai_factory_model.llm.model_utils.load_from_file", autospec=True) as mock_load_from_file:
            # Configurar el mock
            mock_load_from_file.return_value = {
                "test_alias": {
                    "connection_type": "AzureOpenAIChat",
                    "model_name": "test_model",
                    "model_version": "1.0",
                    "api_auth": "service_principal",
                    "api_endpoint": "https://example.com",
                    "model_params": {}
                }
            }

            # Create model instance and add to the cache
            config = mock_load_from_file.return_value["test_alias"]
            model = AzureOpenAIChatModel(config)
            model.initialize_model("test_alias")
            cache["test_alias"] = model

            # Verify get from cache
            retrieved_model = ModelFactory.get_model("test_alias")
            assert retrieved_model == model
    else:
        assert True  # pragma: no cover


@patch("ai_factory_model.llm.model_AzureOpenAIChat.AzureAuthClient", autospec=True)
def test_create_model(mock_azure_auth_client, env_testing):
    if env_testing:

        mock_auth_client_instance = mock_azure_auth_client.return_value
        mock_auth_client_instance.get_token.return_value = "mock_token"

        config = {
            "connection_type": "AzureOpenAIChat",
            "model_name": "test_model",
            "model_version": "1.0",
            "api_auth": "service_principal",
            "api_endpoint": "https://example.com",
            "model_params": {}
        }

        created_model = ModelFactory.create_model("test_alias", config)
        assert created_model.client.azure_endpoint == "https://example.com"
        assert created_model.client.azure_ad_token_provider is not None
        assert created_model.client.deployment_name == "test_model"
        assert created_model.client.openai_api_version == "1.0"
    else:
        assert True  # pragma: no cover
