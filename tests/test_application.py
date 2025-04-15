from langchain_core.messages.ai import AIMessage
from src.ai_factory_model.llm import ModelFactory
from src.ai_factory_model.logger.utils import info


def test_app(env_testing):

    if env_testing:
        # AZURE_OPENAI_CHAT_DEPLOYMENT
        model = ModelFactory.get_model("azai_gtp4o")
        params = ["Eres un guía turístico", "¿Cuál es la capital de Francia?"]
        response = model.prompt(params=params)
        # print(response)
        info(f"{response}")
        assert isinstance(response, str)
    else:
        assert True


def test_aimessage(env_testing):

    if env_testing:
        model = ModelFactory.get_model("azai_gtp4o")
        params = ["Eres un guía turístico", "¿Cuál es la capital de Francia?"]

        response = model.get_client.invoke([
            {"role": "system", "content": params[0]},
            {"role": "user", "content": params[1]}
        ])
        info(f"{response.content}")
        assert isinstance(response, AIMessage)
    else:
        assert True
