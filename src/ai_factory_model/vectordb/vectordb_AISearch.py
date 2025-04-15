from .vectordb_base import BaseVectorDB
from azure.core.credentials import AzureKeyCredential
from azure.core.exceptions import ResourceNotFoundError
from azure.search.documents import SearchClient
from azure.search.documents.models import VectorizedQuery


class AISearchVectorDB(BaseVectorDB):

    def __init__(self, config):
        super().__init__(config)

    def initialize_vectorDB(self, alias):

        self.client = SearchClient(endpoint=self.endpoint,
                                   index_name=self.index_name,
                                   credential=AzureKeyCredential(self.api_key))
        self.alias = alias
        return self

    def get_search_client(self):
        return self.client

    def search_by_key(self, key, fields=None):
        try:
            result = self.client.get_document(key=key,
                                              selected_fields=fields or self.fields)
        except ResourceNotFoundError:
            result = {}
        return result

    def search_by_vector(self,
                         vector,
                         vector_field=None,
                         filters=None,
                         fields=None,
                         limit=10):

        if fields is None:
            fields = self.fields

        if vector_field is None:
            vector_field = self.index_vector

        vector_query = VectorizedQuery(vector=vector, k_nearest_neighbors=limit, fields=vector_field)
        results = self.client.search(
            search_text=None,
            vector_queries=[vector_query],
            select=fields
        )
        return results
