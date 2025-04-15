from ..logger import error
from ..config import MODELS_CONFIG_FILE  # VECTORDB_INDEX, VECTORDB_EMBEDDING_MODEL


from .vectordb_base import BaseVectorDB
from .vectordb_AISearch import AISearchVectorDB
from .vectordb_PGVector import PGVectorDB
from ..llm.model_utils import load_from_file

cache = dict()


class VectorDBFactory:

    VDB_CLASSES = {
        "AISearchVectorDB": AISearchVectorDB,
        "PGVectorDB": PGVectorDB
    }

    @staticmethod
    def get_index(alias: str) -> BaseVectorDB:

        cache_key = "index_" + alias
        cached = cache.get(cache_key)
        if cached is not None:
            return cached

        config = load_from_file(MODELS_CONFIG_FILE)

        try:
            vector_def = config.get(alias)
            if vector_def is not None:
                model_class = VectorDBFactory.VDB_CLASSES.get(vector_def.get("connection_type"))
                vdb: BaseVectorDB = model_class(vector_def)
                vdb.initialize_vectorDB(alias)
                cache[cache_key] = vdb
                return vdb
        except Exception as e:
            error_msg = f"Error in VectorDBFactory.create {e}"
            error(f"{error_msg}")
            raise Exception(error_msg)
