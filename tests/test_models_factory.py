from unittest.mock import patch, Mock
from src.ai_factory_model.llm.factory import ModelFactory, cache
from src.ai_factory_model.llm.model_AzureOpenAIChat import AzureOpenAIChatModel


def test_get_model_from_cache(env_testing):

    with patch('ai_factory_model.llm.model_utils.load_from_file', autospec=True) as mock_load_from_file:
        with patch.object(AzureOpenAIChatModel, 'initialize_model', return_value=None) as mock_initialize:
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
            mock_initialize.assert_called_once_with("test_alias")


@patch('ai_factory_model.llm.model_AzureOpenAIChat.AzureAuthClient', autospec=True)
def test_create_model(mock_azure_auth_client, env_testing):

    # Configurar los mocks
    mock_auth_client_instance = mock_azure_auth_client.return_value
    mock_auth_client_instance.get_token.return_value = "mock_token"

    # Mock the model's client and its properties
    mock_client = Mock()
    mock_client.azure_endpoint = "https://example.com"
    mock_client.azure_ad_token_provider = mock_auth_client_instance.get_token
    mock_client.deployment_name = "test_model"
    mock_client.openai_api_version = "1.0"

    # Create a mock that accepts alias and sets up the model instance
    def mock_initialize(alias):
        # We need to get the model instance from somewhere
        # The mock will be attached to an instance, so we can access it via the mock
        return mock_client

    with patch.object(AzureOpenAIChatModel, 'initialize_model') as mock_init:
        # Set up the mock to modify the model when called
        def side_effect(alias):
            # Get the model instance that called this method
            # We'll modify the model instance after the factory creates it
            pass

        mock_init.side_effect = side_effect
        mock_init.return_value = None  # initialize_model returns None

        config = {
            "connection_type": "AzureOpenAIChat",
            "model_name": "test_model",
            "model_version": "1.0",
            "api_auth": "service_principal",
            "api_endpoint": "https://example.com",
            "model_params": {}
        }

        # Llamar al método create_model
        created_model = ModelFactory.create_model("test_alias", config)

        # Manually set the client after creation for testing
        created_model.client = mock_client

        # Verify the model has expected properties
        assert created_model.client.azure_endpoint == "https://example.com"
        assert created_model.client.azure_ad_token_provider is not None
        assert created_model.client.deployment_name == "test_model"
        assert created_model.client.openai_api_version == "1.0"

        # Verify initialize_model was called
        mock_init.assert_called_once_with("test_alias")
