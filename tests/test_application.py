import pytest
from langchain_core.messages.ai import AIMessage
from src.ai_factory_model.llm import ModelFactory, AzureOpenAIChatModel
from src.ai_factory_model.logger.utils import info


@pytest.fixture
def azure_openaichatmodel(env_testing):
    if env_testing:
        # Only configure model when variables are present
        return AzureOpenAIChatModel(config={
            "connection_type": "AzureOpenAIChat",
            "model_name": "{AZURE_OPENAI_CHAT_DEPLOYMENT}",
            "model_version": "{AZURE_OPENAI_API_VERSION}",
            "api_key": "{AZURE_OPENAPI_KEY}",
            "api_endpoint": "{AZURE_OPENAI_ENDPOINT}",
            "api_auth": "api_key",
            "model_params": {
                "max_tokens": 512,
                "temperature": 0.00000001,
                "frequency_penalty": 0.000005
            }
        })
    else:
        return None


def test_app_service_principal(env_testing):

    if env_testing:
        model = ModelFactory.get_model("azai_gtp4o")
        params = ["Eres un guía turístico", "¿Dónde está Plasencia?"]

        response = model.prompt(params=params)
        info(f"{response}")
        assert isinstance(response, str)
    else:
        assert True


def test_app_api_key(env_testing, azure_openaichatmodel):

    if env_testing:
        model = azure_openaichatmodel
        model.initialize_model("azai_gtp4o")
        params = ["Eres un guía turístico", "¿Dónde está Mérida?"]

        response = model.prompt(params=params)
        info(f"{response}")
        assert isinstance(response, str)
    else:
        assert True


def test_aimessage(env_testing):

    if env_testing:
        model = ModelFactory.get_model("azai_gtp4o")
        params = ["Eres un guía turístico", "¿Cuál es la capital de España?"]

        response = model.get_client.invoke([
            {"role": "system", "content": params[0]},
            {"role": "user", "content": params[1]}
        ])
        info(type(response))
        info(f"{response.content}")
        assert isinstance(response, AIMessage)
    else:
        assert True
