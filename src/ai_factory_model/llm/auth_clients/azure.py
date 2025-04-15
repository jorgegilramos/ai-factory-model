from azure.identity import ClientSecretCredential

from ...config import AZURE_TENANT_ID, AZURE_CLIENT_ID, AZURE_CLIENT_SECRET, \
    AZURE_TOKEN_URL


# Azure Authentication Client
class AzureAuthClient:
    def __init__(self):
        self.credential = ClientSecretCredential(
            tenant_id=AZURE_TENANT_ID,
            client_id=AZURE_CLIENT_ID,
            client_secret=AZURE_CLIENT_SECRET
        )

    def get_token(self):
        return self.credential.get_token(AZURE_TOKEN_URL).token
