from abc import ABC, abstractmethod
import re
from decouple import config

from ..config import get_var, kwargs_decouple

REGEX_VAR = r"{(.+?)}"


class BaseVectorDB(ABC):
    def __init__(self, config: dict[str, str]):

        self.config = config
        self.endpoint = self.render_var("api_endpoint")
        self.fields = self.render_var("index_fields")
        self.index_name = self.render_var("index_name", cast=str)
        self.api_key = self.render_var("api_key", cast=str)
        self.api_auth = self.render_var("api_auth", default="api_key")
        self.index_vector = self.render_var("index_vector")
        self.params = self.render_var("index_params", default={})

    @abstractmethod
    def initialize_vectorDB(self):
        """Implementa la inicialización del modelo de IA."""
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
