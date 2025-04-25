from abc import ABC, abstractmethod
import datetime
import re
from decouple import config
from ..logger import info
from ..config import get_var, kwargs_decouple
from .model_utils import read_template


REGEX_VAR = r"{(.+?)}"

# Label constants
LABEL_MODEL_NAME = "model_name"
LABEL_MODEL_VERSION = "model_version"
LABEL_API_KEY = "api_key"
LABEL_API_ENDPOINT = "api_endpoint"
LABEL_API_AUTH = "api_auth"
LABEL_MODEL_PARAMS = "model_params"

# Value constants
VALUE_SERVICE_PRINCIPAL = "service_principal"
VALUE_API_KEY = "api_key"


class BaseModel(ABC):

    # Create attribute
    auth_client = None
    client = None

    def __init__(self, config: dict[str, str]):
        self.config = config

        self.model_name = self.render_var(LABEL_MODEL_NAME)
        self.version = self.render_var(LABEL_MODEL_VERSION)

        if LABEL_API_ENDPOINT in config:
            self.endpoint = self.render_var(LABEL_API_ENDPOINT)

        # Authentication
        self.api_auth = self.render_var(LABEL_API_AUTH, default=VALUE_SERVICE_PRINCIPAL)
        if self.api_auth == VALUE_SERVICE_PRINCIPAL:
            # Create attribute
            self.auth_client = None
        elif self.api_auth == VALUE_API_KEY and LABEL_API_KEY in config:
            self.api_key = self.render_var(LABEL_API_KEY, cast=str)

        self.params = self.render_var(LABEL_MODEL_PARAMS, default={})

        # Client will be initialized in initialize_model
        self.client = None

    @abstractmethod
    def initialize_model(self):
        """
        Initialization of model AI implementation.
        """
        pass

    @property
    def get_client(self):
        return self.client

    def prompt(self, params) -> str:

        system = params[0]
        input = params[1]

        init = datetime.datetime.now()
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": input}
        ]

        response = self.client.invoke(messages).content
        duration = (datetime.datetime.now() - init).total_seconds()
        info(f"Duration: {duration:.4f} seconds")
        return response

    def prompt_template(self, path, params: dict):
        return self.prompt(read_template(path, params))

    def embedding(self, text: str):
        raise Exception(f"Model {self.model_name} does not allow embedding")

    def render_var(self, var_name, *args, **kwargs):
        """
        Render environment variable using decouple and applying conventions in value as {kv:VARIABLE}
        """
        # Get decouple arguments
        decouple_kwargs = kwargs_decouple(*args, **kwargs)
        # Property has to exist
        if var_name in self.config:
            property_value: str = self.config.get(var_name)
            if isinstance(property_value, str):
                # Only render string templates
                return self.render_property(property_value, **decouple_kwargs)
            return property_value
        else:
            # Get value using decouple, will throw a ValueError if the variable doesn't exist
            # and it is not configurated a default value
            return config(var_name, **decouple_kwargs)

    def render_property(self, property_value: str, **decouple_kwargs) -> str:
        """
        Render property value or template
        """
        values_list: dict[str, str] = {}
        var_names = re.findall(REGEX_VAR, property_value)
        if var_names:
            # Template found
            for var_name in var_names:
                # Add values from environment variables using decouple
                values_list[var_name] = get_var(var_name=var_name, **decouple_kwargs)
        value = property_value.format(**values_list)
        return value
