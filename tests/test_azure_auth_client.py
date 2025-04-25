import pytest
from unittest.mock import patch, MagicMock
from src.ai_factory_model.llm.auth_clients import AzureAuthClient
# from src.ai_factory_model.logger.utils import info


@pytest.fixture
def azure_auth_client(env_testing):
    if env_testing:
        # Only configure model when variables are present
        return AzureAuthClient()
    else:
        return None


def test_should_refresh_token_not_none(env_testing, azure_auth_client):
    if env_testing:
        with patch("time.time", return_value=1000):
            azure_auth_client.token = MagicMock(refresh_on=900, expires_on=1300)
            assert azure_auth_client._should_refresh() is True
    else:
        assert True


def test_should_not_refresh_token_not_expired(env_testing, azure_auth_client):
    if env_testing:
        with patch("time.time", return_value=1000):
            azure_auth_client.token = MagicMock(refresh_on=1100, expires_on=1400)
            assert azure_auth_client._should_refresh() is False
    else:
        assert True


def test_should_not_refresh_recent_request(env_testing, azure_auth_client):
    if env_testing:
        with patch("time.time", return_value=1000):
            azure_auth_client.token = MagicMock(refresh_on=1100, expires_on=1300)
            azure_auth_client._last_request_time = 980
            assert azure_auth_client._should_refresh() is False
    else:
        assert True


def test_get_token_refresh(env_testing, azure_auth_client):
    if env_testing:
        with patch("time.time", return_value=1000):
            with patch.object(azure_auth_client.credential,
                              "get_token_info",
                              return_value=MagicMock(token="new_token")):
                azure_auth_client.token = MagicMock(refresh_on=900, expires_on=1300)
                token = azure_auth_client.get_token()
                assert token == "new_token"
                assert azure_auth_client._last_request_time == 1000
    else:
        assert True


def test_get_token_no_refresh(env_testing, azure_auth_client):
    if env_testing:
        with patch("time.time", return_value=1000):
            azure_auth_client._last_request_time = 980
            azure_auth_client.token = MagicMock(token="existing_token", refresh_on=1100, expires_on=1300)
            token = azure_auth_client.get_token()
            assert token == "existing_token"
            assert azure_auth_client._last_request_time == 980
    else:
        assert True


def test_get_token_provider(env_testing, azure_auth_client):
    if env_testing:
        token_provider = azure_auth_client.get_token_provider()
        assert token_provider == azure_auth_client.token_provider
    else:
        assert True
