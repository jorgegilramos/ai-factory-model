from .vectordb_base import BaseVectorDB

from logging import error
import os
import psycopg

# https://python.langchain.com/docs/integrations/vectorstores/pgvector/
# https://github.com/langchain-ai/langchain-postgres/blob/main/examples/vectorstore.ipynb


class PGVectorDB(BaseVectorDB):

    def __init__(self, config):
        self.index_key = config.get("index_key").format(**os.environ)
        super().__init__(config)

    def initialize_vectorDB(self, alias):
        # obtener la conexión a la base de datos
        self.client = psycopg.connect(self.endpoint)
        self.alias = alias
        return self

    def exec_st(self, st: str, args):
        cur = None
        try:
            cur = self.client.cursor()
            cur.execute(st, args)
            result = cur.fetchall()
        except Exception as e:
            error(f"{e}")
        finally:
            if cur is None:
                cur.close()
        return result

    def search_by_key(self, key_value, fields):

        if fields is None:
            fields = self.fields

        st = """ SELECT "%s", 1 as "@search.score" FROM %s where "%s" = '%s'""" \
             % ("\" , \"".join(fields), self.index_name, self.index_key, "%s")

        result = self.exec_st(st, (self.key_value,))
        return self.results_to_dict(result, fields)

    def results_to_dict(self, results, fields: list) -> dict:
        # Create dictionaries with only the selected fields
        return [{fields[i]: r[i] for i in range(0, len(r))} for r in results]

    def search_by_vector(self,
                         vector,
                         vector_field=None,
                         filters=None,
                         fields=None,
                         limit=10):

        if fields is None:
            fields = self.fields

        st_where = ""
        if filters is not None and filters.get("where") is not None:
            st_where = " WHERE " + filters.get("where")

        if vector_field is None:
            vector_field = self.index_vector

        st = """
        SELECT "%s", 1 - ("%s" <=> %s::vector) as "@search.score"
        FROM %s %s order by "@search.score" desc Limit %s
        """ \
        % ("\" , \"".join(fields), vector_field, "%s", self.index_name, st_where, limit)
        return self.search_by_statement(st, vector, fields + ["@search.score"])

    def search_by_statement(self, st, vector, fields=[]):
        result = self.exec_st(st, (vector,))
        return self.results_to_dict(result, fields)
