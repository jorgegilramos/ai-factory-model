import pytest
from unittest.mock import patch, Mock
from azure.core.exceptions import HttpResponseError
from src.ai_factory_model.security.keyvault_handler import KeyVaultHandler


class TestKeyVaultHandler:
    """Test suite for KeyVaultHandler class."""

    def setup_method(self):
        """Set up test fixtures before each test method."""
        self.kv_name = "test-keyvault"
        self.kv_tenant_id = "test-tenant-id"
        self.kv_client_id = "test-client-id"
        self.kv_secret = "test-secret"
        self.custom_kv_url = "https://custom-vault.vault.azure.net"
        self.authority = "https://login.microsoftonline.com"

    @patch('src.ai_factory_model.security.keyvault_handler.SecretClient')
    @patch('src.ai_factory_model.security.keyvault_handler.ClientSecretCredential')
    def test_init_with_default_url(self, mock_credential, mock_secret_client):
        """Test KeyVaultHandler initialization with default URL."""
        # Arrange
        mock_credential_instance = Mock()
        mock_credential.return_value = mock_credential_instance
        mock_client_instance = Mock()
        mock_secret_client.return_value = mock_client_instance

        # Act
        handler = KeyVaultHandler(
            kv_name=self.kv_name,
            kv_tenant_id=self.kv_tenant_id,
            kv_client_id=self.kv_client_id,
            kv_secret=self.kv_secret
        )

        # Assert
        assert handler.kv_name == self.kv_name
        assert handler.kv_client_id == self.kv_client_id
        assert handler.kv_secret == self.kv_secret
        assert handler.kv_tenant_id == self.kv_tenant_id
        assert handler.kv_url == f"https://{self.kv_name}.vault.azure.net"
        assert handler.authority == self.authority
        assert handler.kv_client == mock_client_instance

        # Verify credential creation
        mock_credential.assert_called_once_with(
            tenant_id=self.kv_tenant_id,
            client_id=self.kv_client_id,
            client_secret=self.kv_secret,
            authority=self.authority
        )

        # Verify client creation
        mock_secret_client.assert_called_once_with(
            f"https://{self.kv_name}.vault.azure.net",
            mock_credential_instance
        )

    @patch('src.ai_factory_model.security.keyvault_handler.SecretClient')
    @patch('src.ai_factory_model.security.keyvault_handler.ClientSecretCredential')
    def test_init_with_custom_url_and_authority(self, mock_credential, mock_secret_client):
        """Test KeyVaultHandler initialization with custom URL and authority."""
        # Arrange
        custom_authority = "https://custom.login.com"
        mock_credential_instance = Mock()
        mock_credential.return_value = mock_credential_instance
        mock_client_instance = Mock()
        mock_secret_client.return_value = mock_client_instance

        # Act
        handler = KeyVaultHandler(
            kv_name=self.kv_name,
            kv_tenant_id=self.kv_tenant_id,
            kv_client_id=self.kv_client_id,
            kv_secret=self.kv_secret,
            kv_url=self.custom_kv_url,
            authority=custom_authority
        )

        # Assert
        assert handler.kv_url == self.custom_kv_url
        assert handler.authority == custom_authority

        # Verify credential creation with custom authority
        mock_credential.assert_called_once_with(
            tenant_id=self.kv_tenant_id,
            client_id=self.kv_client_id,
            client_secret=self.kv_secret,
            authority=custom_authority
        )

        # Verify client creation with custom URL
        mock_secret_client.assert_called_once_with(
            self.custom_kv_url,
            mock_credential_instance
        )

    @patch('src.ai_factory_model.security.keyvault_handler.SecretClient')
    @patch('src.ai_factory_model.security.keyvault_handler.ClientSecretCredential')
    def test_connect_client(self, mock_credential, mock_secret_client):
        """Test connect_client method."""
        # Arrange
        mock_credential_instance = Mock()
        mock_credential.return_value = mock_credential_instance
        mock_client_instance = Mock()
        mock_secret_client.return_value = mock_client_instance

        # Act
        handler = KeyVaultHandler(
            kv_name=self.kv_name,
            kv_tenant_id=self.kv_tenant_id,
            kv_client_id=self.kv_client_id,
            kv_secret=self.kv_secret
        )

        # Call connect_client again to test it directly
        handler.connect_client()

        # Assert - verify it was called twice (once in __init__, once directly)
        assert mock_credential.call_count == 2
        assert mock_secret_client.call_count == 2

    @patch('src.ai_factory_model.security.keyvault_handler.SecretClient')
    @patch('src.ai_factory_model.security.keyvault_handler.ClientSecretCredential')
    def test_get_secret_success(self, mock_credential, mock_secret_client):
        """Test get_secret method success case."""
        # Arrange
        secret_name = "test-secret-name"
        secret_value = "test-secret-value"

        mock_credential.return_value = Mock()
        mock_client_instance = Mock()
        mock_secret_client.return_value = mock_client_instance

        # Mock the secret object
        mock_secret = Mock()
        mock_secret.value = secret_value
        mock_client_instance.get_secret.return_value = mock_secret

        # Act
        handler = KeyVaultHandler(
            kv_name=self.kv_name,
            kv_tenant_id=self.kv_tenant_id,
            kv_client_id=self.kv_client_id,
            kv_secret=self.kv_secret
        )
        result = handler.get_secret(secret_name)

        # Assert
        assert result == secret_value
        mock_client_instance.get_secret.assert_called_with(secret_name)

    @patch('src.ai_factory_model.security.keyvault_handler.error')
    @patch('src.ai_factory_model.security.keyvault_handler.SecretClient')
    @patch('src.ai_factory_model.security.keyvault_handler.ClientSecretCredential')
    def test_get_secret_http_response_error(self, mock_credential, mock_secret_client, mock_error):
        """Test get_secret method with HttpResponseError."""
        # Arrange
        secret_name = "test-secret-name"
        http_error = HttpResponseError("Key vault error")

        mock_credential.return_value = Mock()
        mock_client_instance = Mock()
        mock_secret_client.return_value = mock_client_instance
        mock_client_instance.get_secret.side_effect = http_error

        # Act & Assert
        handler = KeyVaultHandler(
            kv_name=self.kv_name,
            kv_tenant_id=self.kv_tenant_id,
            kv_client_id=self.kv_client_id,
            kv_secret=self.kv_secret
        )

        with pytest.raises(HttpResponseError):
            handler.get_secret(secret_name)

        # Verify error logging
        assert mock_error.call_count == 2
        mock_error.assert_any_call("Keyvault not responding")
        mock_error.assert_any_call(f"Error: {http_error}")

    @patch('src.ai_factory_model.security.keyvault_handler.error')
    @patch('src.ai_factory_model.security.keyvault_handler.SecretClient')
    @patch('src.ai_factory_model.security.keyvault_handler.ClientSecretCredential')
    def test_get_secret_general_exception(self, mock_credential, mock_secret_client, mock_error):
        """Test get_secret method with general Exception."""
        # Arrange
        secret_name = "test-secret-name"
        general_error = ValueError("Some unexpected error")

        mock_credential.return_value = Mock()
        mock_client_instance = Mock()
        mock_secret_client.return_value = mock_client_instance
        mock_client_instance.get_secret.side_effect = general_error

        # Act & Assert
        handler = KeyVaultHandler(
            kv_name=self.kv_name,
            kv_tenant_id=self.kv_tenant_id,
            kv_client_id=self.kv_client_id,
            kv_secret=self.kv_secret
        )

        with pytest.raises(ValueError):
            handler.get_secret(secret_name)

        # Verify error logging
        mock_error.assert_called_once_with(f"Error: {general_error}")

    @patch('src.ai_factory_model.security.keyvault_handler.debug')
    @patch('src.ai_factory_model.security.keyvault_handler.SecretClient')
    @patch('src.ai_factory_model.security.keyvault_handler.ClientSecretCredential')
    def test_exist_secret_true(self, mock_credential, mock_secret_client, mock_debug):
        """Test exist_secret method returning True."""
        # Arrange
        secret_name = "existing-secret"

        mock_credential.return_value = Mock()
        mock_client_instance = Mock()
        mock_secret_client.return_value = mock_client_instance
        mock_client_instance.get_secret.return_value = Mock()  # Success

        # Act
        handler = KeyVaultHandler(
            kv_name=self.kv_name,
            kv_tenant_id=self.kv_tenant_id,
            kv_client_id=self.kv_client_id,
            kv_secret=self.kv_secret
        )
        result = handler.exist_secret(secret_name)

        # Assert
        assert result is True
        mock_client_instance.get_secret.assert_called_with(secret_name)
        mock_debug.assert_called_once_with(f"Checking secret name: {secret_name}")

    @patch('src.ai_factory_model.security.keyvault_handler.error')
    @patch('src.ai_factory_model.security.keyvault_handler.debug')
    @patch('src.ai_factory_model.security.keyvault_handler.SecretClient')
    @patch('src.ai_factory_model.security.keyvault_handler.ClientSecretCredential')
    def test_exist_secret_false(self, mock_credential, mock_secret_client, mock_debug, mock_error):
        """Test exist_secret method returning False."""
        # Arrange
        secret_name = "non-existing-secret"
        test_exception = Exception("Secret not found")

        mock_credential.return_value = Mock()
        mock_client_instance = Mock()
        mock_secret_client.return_value = mock_client_instance
        mock_client_instance.get_secret.side_effect = test_exception

        # Act
        handler = KeyVaultHandler(
            kv_name=self.kv_name,
            kv_tenant_id=self.kv_tenant_id,
            kv_client_id=self.kv_client_id,
            kv_secret=self.kv_secret
        )
        result = handler.exist_secret(secret_name)

        # Assert
        assert result is False
        mock_client_instance.get_secret.assert_called_with(secret_name)
        mock_debug.assert_called_once_with(f"Checking secret name: {secret_name}")
        mock_error.assert_called_once_with(f"Error: {test_exception}")
