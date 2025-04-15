import pytest
from decouple import config

# Keyvault variables
KV_NAME: str = config("KV_NAME", None)


@pytest.fixture(scope="function")
def env_testing():

    return KV_NAME is not None
