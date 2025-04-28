from unittest.mock import MagicMock, patch
from src.ai_factory_model.llm.model_AzureOpenAIChat import AzureAuthClient


@patch("src.ai_factory_model.llm.auth_clients.azure.AzureAuthClient._create_credential")
@patch("src.ai_factory_model.llm.auth_clients.azure.AzureAuthClient._create_token_provider")
def test_initialization(mock_create_credential, mock_create_token_provider):

    auth_client = AzureAuthClient()

    assert isinstance(auth_client, AzureAuthClient)


@patch("src.ai_factory_model.llm.auth_clients.azure.AzureAuthClient._create_credential")
@patch("src.ai_factory_model.llm.auth_clients.azure.AzureAuthClient._create_token_provider")
def test_get_token(mock_create_credential, mock_create_token_provider):

    mock_token = MagicMock()
    mock_token.token = "mock_token"

    auth_client = AzureAuthClient()
    auth_client.credential.get_token_info.return_value = mock_token

    # Get token
    token = auth_client.get_token()

    assert token is not None
    assert token == "mock_token"
