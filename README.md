# ai-factory-model


[![PyPI version](https://img.shields.io/pypi/v/ai-factory-model.svg)](https://pypi.org/project/ai-factory-model/)
![Build Status](https://github.com/jorgegilramos/ai-factory-model/workflows/Python%20package/badge.svg)
![PyPI - Downloads](https://img.shields.io/pypi/dm/ai-factory-model)



**ai-factory-model** is a modular Python library aimed at integrating with multiple language models (LLMs), cloud providers, and auxiliary utilities for development and infrastructure.

This library is designed to facilitate interaction with LLMs from OpenAI, Azure, Google, and Ollama, also integrating authentication, external configuration, Jinja2 templates, and reusable components.

## Features

- Support for multiple LLM models:
  - OpenAI (chat and embeddings)
  - Azure OpenAI
  - Google Generative AI
  - Ollama
  - LangChain and variants
- Configuration modules (`decouple`, `YAML`)
- Authentication via Azure Identity
- Content generation via Jinja2 templates
- Clear separation of responsibilities with modules such as:
  - `logger`
  - `security`
  - `auth_clients`
  - `model_*`  (interfaces for different LLMs)

## Installation

From PyPI:

```bash
pip install ai-factory-model
```


## Setup
To use the model factory, you need to define a series of environment variables that allow connection to the various model hosting services:

```python
AZURE_TENANT_ID = <id_tenant_azure>
AZURE_CLIENT_ID = <id_client_azure>
AZURE_CLIENT_SECRET = <secret_passphrase_azure_client>
AZURE_TOKEN_URL = <azure_url_token_generator>
```

For enhanced security, there is a connection to KeyVault. To define the connection to the corresponding key store, use:
```python
KV_NAME = <kv_name>
KV_TENANT_ID = <id_kv_tenant>
KV_CLIENT_ID = <id_kv_client>
KV_SECRET = <secret_passphrase_kv>
```

With the KeyVault connection established, the values to be retrieved from the key store should be specified using the following nomenclature:

> VARIABLE_SECRET = kv{name-of-secret-at-kv}

For example:
```python
# Transition from having the secret in raw form
AZURE_CLIENT_SECRET = <secret_passphrase_azure_client>

# To retrieving it from the KV
AZURE_CLIENT_SECRET = kv{<name_secret_azure_client>}
```

Additionally, if you have a file containing the various model configurations you wish to use, you should specify it with the corresponding variable.

> MODELS_CONFIG_FILE = <path_to_models_declarations_file>

## Basic usage

Using prompt:
```python
from factory_model import ModelFactory

model = ModelFactory.get_model("azai_gtp4o")
params = ["Eres un guía turístico", "¿Dónde está Plasencia?"]

response = model.prompt(params=params)

print(response)
# Output:
# Plasencia es una ciudad situada en la comunidad autónoma de Extremadura, en el oeste de España. Se encuentra en la provincia de Cáceres, a orillas del río Jerte. Plasencia está aproximadamente a unos 80 kilómetros al norte de la ciudad de Cáceres y a unos 250 kilómetros al oeste de Madrid. Es conocida por su casco histórico, que incluye la Catedral de Plasencia, y por su cercanía al Valle del Jerte, famoso por sus cerezos en flor.

```


Using langchain instance:
```python
from factory_model import ModelFactory
params = ["Eres un guía turístico", "¿Cuál es la capital de España?"]

response = model.get_client.invoke([
    {"role": "system", "content": params[0]},
    {"role": "user", "content": params[1]}
])

print(type(response))
# Output:
# <class 'langchain_core.messages.ai.AIMessage'>

print(f"{response.content}")
# Output:
# La capital de España es Madrid. Es una ciudad vibrante y llena de historia, conocida por su rica cultura, su arquitectura impresionante y su animada vida nocturna. Además, Madrid alberga importantes museos como el Museo del Prado y el Museo Reina Sofía, así como el Palacio Real y el Parque del Retiro.
```


## Project structure

```
factory_model/
├── factory_model/
│   ├── __init__.py
│   ├── config/
│   ├── llm/
│   ├── logger/
│   ├── security/
│   ├── vectordb/
├── pyproject.toml
├── requirements.txt
└── README.md
```

## Dependencies

This package requires the followind external libraries:

- `python-decouple`
- `PyYAML`
- `openai`
- `jinja2`
- `azure-core`
- `azure-identity`
- `azure-keyvault-secrets`
- `langchain`
- `langchain-openai`
- `langchain-google-genai`
- `langchain-community`
- `langchain-azure-ai`
- `langchain-ollama`
- `langchain-cohere`
- `azure-search-documents`
- `psycopg[binary]`


## Requirements
- Python 3.12 o superior
- Credential access/API keys to your needed providers (OpenAI, Azure, etc.)
