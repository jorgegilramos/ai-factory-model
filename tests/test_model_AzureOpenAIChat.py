import unittest
from unittest.mock import patch
from assertpy import assert_that, fail


from src.ai_factory_model.llm import AzureOpenAIChatModel
from src.ai_factory_model.llm.model_AzureOpenAIChat import AzureChatOpenAI


class TestAzureOpenAIChatModel(unittest.TestCase):

    @patch('ai_factory_model.llm.model_AzureOpenAIChat.AzureAuthClient', autospec=True)
    @patch('ai_factory_model.llm.model_AzureOpenAIChat.AzureChatOpenAI', autospec=True)
    def test_initialize_model_service_principal(self, mock_azure_chat_openai, mock_azure_auth_client):
        # Configurar los mocks
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

        # Realizar las aserciones individuales
        assert_that(model.client.azure_endpoint).is_equal_to("https://example.com")
        assert_that(model.client.azure_ad_token).is_not_none()
        assert_that(model.client.deployment_name).is_equal_to("test_model")
        assert_that(model.client.openai_api_version).is_equal_to("1.0")
        assert_that(model.client).is_instance_of(AzureChatOpenAI)

    @patch('ai_factory_model.llm.model_AzureOpenAIChat.AzureChatOpenAI', autospec=True)
    def test_initialize_model_api_key(self, mock_azure_chat_openai):
        # Configurar el mock

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

        # Realizar las aserciones individuales
        assert_that(model.client.azure_endpoint).is_equal_to("https://example.com")
        assert_that(model.client.openai_api_key).is_not_none()
        assert_that(model.client.deployment_name).is_equal_to("test_model")
        assert_that(model.client.openai_api_version).is_equal_to("1.0")
        assert_that(model.client).is_instance_of(AzureChatOpenAI)

    def test_initialize_model_invalid_auth(self):
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
            fail("Expected ValueError to be raised")
        except ValueError as e:
            assert_that(str(e)).is_equal_to('Authorization should be "service_principal" or "api_key"')

    @patch('ai_factory_model.llm.model_AzureOpenAIChat.AzureAuthClient', autospec=True)
    @patch('ai_factory_model.llm.model_AzureOpenAIChat.AzureChatOpenAI', autospec=True)
    def test_create_model(self, mock_azure_chat_openai, mock_azure_auth_client):
        # Configurar los mocks
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

        # Realizar las aserciones individuales
        assert_that(model.client.azure_endpoint).is_equal_to("https://example.com")
        assert_that(model.client.azure_ad_token).is_not_none()
        assert_that(model.client.deployment_name).is_equal_to("test_model")
        assert_that(model.client.openai_api_version).is_equal_to("1.0")
        assert_that(model.client).is_instance_of(AzureChatOpenAI)

    @patch('ai_factory_model.llm.model_AzureOpenAIChat.AzureChatOpenAI', autospec=True)
    def test_read_model(self, mock_azure_chat_openai):
        # Configurar el mock
        mock_client_instance = mock_azure_chat_openai.return_value

        config = {
            "model_name": "test_model",
            "model_version": "1.0",
            "api_auth": "service_principal",
            "api_endpoint": "https://example.com",
            "model_params": {}
        }

        model = AzureOpenAIChatModel(config)
        model.client = mock_client_instance  # Asegurarse de que el cliente se inicializa correctamente
        client = model.get_client

        # Realizar las aserciones
        assert_that(client).is_equal_to(mock_client_instance)

    @patch('ai_factory_model.llm.model_AzureOpenAIChat.AzureChatOpenAI', autospec=True)
    def test_update_model(self, mock_azure_chat_openai):
        # Configurar el mock

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

        # Realizar las aserciones
        assert_that(model.model_name).is_equal_to("updated_model")

    @patch('ai_factory_model.llm.model_AzureOpenAIChat.AzureChatOpenAI', autospec=True)
    def test_delete_model(self, mock_azure_chat_openai):
        # Configurar el mock
        config = {
            "model_name": "test_model",
            "model_version": "1.0",
            "api_auth": "service_principal",
            "api_endpoint": "https://example.com",
            "model_params": {}
        }

        model = AzureOpenAIChatModel(config)
        model.client = None

        # Realizar las aserciones
        assert_that(model.client).is_none()


if __name__ == '__main__':
    unittest.main()
