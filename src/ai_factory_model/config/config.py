from decouple import config as _config
import re
from ..security import KeyVaultHandler

# Keyvault variables
KV_NAME: str = _config("KV_NAME", None)
KV_TENANT_ID: str = _config("KV_TENANT_ID", None)
KV_CLIENT_ID: str = _config("KV_CLIENT_ID", None)
KV_SECRET: str = _config("KV_SECRET", None)

REGEX_KV_MATCH = r"^kv{(.+)}"

kv_handler = KeyVaultHandler(
    KV_NAME,
    KV_TENANT_ID,
    KV_CLIENT_ID,
    KV_SECRET
) if KV_NAME is not None else None


def kwargs_decouple(*args, **kwargs):
    config_kwargs = {}
    # Assign well known position arguments
    if len(args) > 0:
        config_kwargs["default"] = args[0]
        if len(args) > 1:
            config_kwargs["cast"] = args[1]
    # Assign well known named arguments
    properties = ["default", "cast"]
    for prop in properties:
        if prop in kwargs:
            config_kwargs[prop] = kwargs[prop]
    return config_kwargs


def get_var(var_name: str, *args, **kwargs):
    # Get decouple arguments
    decouple_kwargs = kwargs_decouple(*args, **kwargs)

    # Get value using decouple
    value = _config(var_name, **decouple_kwargs)

    # Check if it is a keyvault key
    if isinstance(value, str):
        match = re.match(REGEX_KV_MATCH, value)
        if match:
            kv_key = match.groups()[0]
            # Only supported azure keyvault to keep secrets, extend to others when used
            kv: KeyVaultHandler = kwargs["kv"] if "kv" in kwargs else kv_handler
            value = kv.get_secret(secret_name=kv_key)
    return value


AZURE_TENANT_ID = get_var("AZURE_TENANT_ID", None)
AZURE_CLIENT_ID = get_var("AZURE_CLIENT_ID", None)
AZURE_CLIENT_SECRET = get_var("AZURE_CLIENT_SECRET", None)
AZURE_TOKEN_URL = get_var("AZURE_TOKEN_URL", default="https://cognitiveservices.azure.com/.default")

# These variables are not necessary as are referenced by nomenclature
# AZURE_INFERENCE_CREDENTIAL = get_var("AZURE_INFERENCE_CREDENTIAL", "")
# AZURE_INFERENCE_ENDPOINT = get_var("AZURE_INFERENCE_ENDPOINT", "")
# GOOGLE_API_KEY = get_var("GOOGLE_API_KEY", "")

MODELS_CONFIG_FILE = get_var("MODELS_CONFIG_FILE", "./src/params/params.json")

# # Variables vector databases
# VECTORDB_INDEX = get_var("VECTORDB_INDEX", None)
# VECTORDB_EMBEDDING_MODEL = get_var("VECTORDB_EMBEDDING_MODEL", None)
