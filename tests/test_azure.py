from unittest.mock import MagicMock, patch
from src.ai_factory_model.llm.model_AzureOpenAIChat import AzureAuthClient


def test_initialization(env_testing):
    if env_testing:
        # Crear instancia de AzureAuthClient
        auth_client = AzureAuthClient()
        # Verificar que la instancia se crea correctamente
        assert isinstance(auth_client, AzureAuthClient)
    else:
        assert True


@patch('azure.identity.ClientSecretCredential')
def test_get_token(mock_client_secret_credential, env_testing):
    # Verificar que el entorno de prueba está configurado
    if env_testing:
        # Configurar el mock
        mock_credential_instance = mock_client_secret_credential.return_value
        mock_token = MagicMock()
        mock_token.token = "mock_token"
        mock_credential_instance.get_token.return_value = mock_token

        # Crear instancia de AzureAuthClient
        auth_client = AzureAuthClient()

        # Obtener el token
        token = auth_client.get_token()

        # Verificar que el token no es nulo ni está vacío
        assert token is not None
        assert token != ""
    else:
        assert True
