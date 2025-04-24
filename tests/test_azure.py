import unittest
from unittest.mock import patch, MagicMock
from src.ai_factory_model.llm.model_AzureOpenAIChat import AzureAuthClient


class TestAzureAuthClient(unittest.TestCase):

    @patch('azure.identity.ClientSecretCredential', autospec=True)
    def test_initialization(self, mock_client_secret_credential):
        # Crear instancia de AzureAuthClient
        auth_client = AzureAuthClient()
        # Verificar que la instancia se crea correctamente
        self.assertIsInstance(auth_client, AzureAuthClient)

    @patch('azure.identity.ClientSecretCredential', autospec=True)
    def test_get_token(self, mock_client_secret_credential):
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
        self.assertIsNotNone(token)
        self.assertNotEqual(token, "")


if __name__ == '__main__':
    unittest.main()
