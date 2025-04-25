from abc import ABC, abstractmethod
import re
from decouple import config

from ..config import get_var, kwargs_decouple

REGEX_VAR = r"{(.+?)}"

# Label constants
LABEL_MODEL_NAME = "model_name"
LABEL_MODEL_VERSION = "model_version"
LABEL_API_KEY = "api_key"
LABEL_API_ENDPOINT = "api_endpoint"
LABEL_API_AUTH = "api_auth"
LABEL_INDEX_FIELDS = "index_fields"
LABEL_INDEX_NAME = "index_name"
LABEL_INDEX_VECTOR = "index_vector"
LABEL_MODEL_PARAMS = "model_params"

# Value constants
VALUE_SERVICE_PRINCIPAL = "service_principal"
VALUE_API_KEY = "api_key"


class BaseVectorDB(ABC):

    # Create attribute
    auth_client = None
    client = None

    def __init__(self, config: dict[str, str]):

        self.config = config

        self.endpoint = self.render_var(LABEL_API_ENDPOINT)
        self.fields = self.render_var(LABEL_INDEX_FIELDS)
        self.index_name = self.render_var(LABEL_INDEX_NAME, cast=str)

        # Authentication
        self.api_auth = self.render_var(LABEL_API_AUTH, default=VALUE_API_KEY)
        if self.api_auth == VALUE_SERVICE_PRINCIPAL:
            # Create attribute
            self.auth_client = None
        elif self.api_auth == VALUE_API_KEY and LABEL_API_KEY in config:
            self.api_key = self.render_var(LABEL_API_KEY, cast=str)

        self.index_vector = self.render_var(LABEL_INDEX_VECTOR)
        self.params = self.render_var(LABEL_MODEL_PARAMS, default={})

    @abstractmethod
    def initialize_vectorDB(self):
        """
        Initialization of model AI implementation.
        """
        pass

    def get_search_client(self):
        None

    def search_by_key(self, key, fields=None):
        None

    def search_by_vector(self,
                         vector,
                         vector_field=None,
                         filters=None,
                         fields=None,
                         limit=10):
        None

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
