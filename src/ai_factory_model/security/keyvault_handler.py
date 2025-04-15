from azure.keyvault.secrets import SecretClient
from azure.identity import ClientSecretCredential
from azure.core.exceptions import HttpResponseError

from ..logger import error, debug


class KeyVaultHandler:

    def __init__(self, kv_name: str,
                 kv_tenant_id: str,
                 kv_client_id: str,
                 kv_secret: str,
                 kv_url: str = None,
                 authority: str = "https://login.microsoftonline.com"):

        self.kv_name = kv_name
        self.kv_client_id = kv_client_id
        self.kv_secret = kv_secret
        self.kv_tenant_id = kv_tenant_id
        self.kv_url = f"https://{kv_name}.vault.azure.net" if kv_url is None else kv_url
        self.authority = authority

        self.connect_client()

    def connect_client(self):
        credential = ClientSecretCredential(
            tenant_id=self.kv_tenant_id,
            client_id=self.kv_client_id,
            client_secret=self.kv_secret,
            authority=self.authority)

        self.kv_client = SecretClient(self.kv_url, credential)

    def get_secret(self, secret_name: str) -> str:
        debug(f"Get secret name: {secret_name}")
        try:
            secret = self.kv_client.get_secret(secret_name).value
        except HttpResponseError as e:
            error("Keyvault not responding")
            error(f"Error: {e}")
            raise e
        except Exception as e:
            error(f"Error: {e}")
            raise e
        return secret

    def exist_secret(self, secret_name) -> bool:
        debug(f"Checking secret name: {secret_name}")
        try:
            self.kv_client.get_secret(secret_name)
            return True
        except Exception as e:
            error(f"Error: {e}")
        return False
