from unittest.mock import patch, Mock
import pytest
from src.ai_factory_model.vectordb.factory import VectorDBFactory, cache
from src.ai_factory_model.vectordb.vectordb_AISearch import AISearchVectorDB
from src.ai_factory_model.vectordb.vectordb_PGVector import PGVectorDB


class TestVectorDBFactory:

    def setup_method(self):
        """Clear cache before each test."""
        cache.clear()

    def test_get_index_from_cache(self):
        """Test retrieving vector database from cache."""
        with patch('src.ai_factory_model.vectordb.factory.load_from_file', autospec=True) as mock_load_from_file:
            # Setup configuration
            mock_load_from_file.return_value = {
                "test_alias": {
                    "connection_type": "AISearchVectorDB",
                    "api_endpoint": "https://test.search.windows.net",
                    "api_key": "test_api_key",
                    "index_name": "test_index",
                    "index_fields": ["id", "content", "vector"],
                    "index_vector": "vector"
                }
            }

            # Create a mock vector database and add to cache
            mock_vdb = Mock(spec=AISearchVectorDB)
            cache_key = "index_test_alias"
            cache[cache_key] = mock_vdb

            # Call get_index and verify it returns cached instance
            result = VectorDBFactory.get_index("test_alias")

            assert result == mock_vdb
            # Verify that load_from_file was not called since we used cache
            mock_load_from_file.assert_not_called()

    def test_get_index_create_aisearch_vectordb(self):
        """Test creating new AISearchVectorDB instance."""
        with patch('src.ai_factory_model.vectordb.factory.load_from_file', autospec=True) as mock_load_from_file:
            with patch.object(AISearchVectorDB, '__init__', return_value=None) as mock_init:
                with patch.object(AISearchVectorDB, 'initialize_vectorDB') as mock_initialize:
                    # Setup configuration
                    config = {
                        "test_alias": {
                            "connection_type": "AISearchVectorDB",
                            "api_endpoint": "https://test.search.windows.net",
                            "api_key": "test_api_key",
                            "index_name": "test_index",
                            "index_fields": ["id", "content", "vector"],
                            "index_vector": "vector"
                        }
                    }
                    mock_load_from_file.return_value = config

                    # Call get_index
                    result = VectorDBFactory.get_index("test_alias")

                    # Verify calls
                    mock_load_from_file.assert_called_once()
                    mock_init.assert_called_once_with(config["test_alias"])
                    mock_initialize.assert_called_once_with("test_alias")

                    # Verify result is cached
                    cache_key = "index_test_alias"
                    assert cache_key in cache
                    assert cache[cache_key] == result

    def test_get_index_create_pgvector_vectordb(self):
        """Test creating new PGVectorDB instance."""
        with patch('src.ai_factory_model.vectordb.factory.load_from_file', autospec=True) as mock_load_from_file:
            with patch.object(PGVectorDB, '__init__', return_value=None) as mock_init:
                with patch.object(PGVectorDB, 'initialize_vectorDB') as mock_initialize:
                    # Setup configuration
                    config = {
                        "test_alias": {
                            "connection_type": "PGVectorDB",
                            "api_endpoint": "postgresql://user:pass@localhost/db",
                            "index_name": "test_table",
                            "index_fields": ["id", "content", "vector"],
                            "index_vector": "embedding"
                        }
                    }
                    mock_load_from_file.return_value = config

                    # Call get_index
                    result = VectorDBFactory.get_index("test_alias")

                    # Verify calls
                    mock_load_from_file.assert_called_once()
                    mock_init.assert_called_once_with(config["test_alias"])
                    mock_initialize.assert_called_once_with("test_alias")

                    # Verify result is cached
                    cache_key = "index_test_alias"
                    assert cache_key in cache
                    assert cache[cache_key] == result

    def test_get_index_missing_configuration(self):
        """Test behavior when configuration is missing for alias."""
        with patch('src.ai_factory_model.vectordb.factory.load_from_file', autospec=True) as mock_load_from_file:
            # Setup empty configuration
            mock_load_from_file.return_value = {}

            # Call get_index - should return None when alias not found
            result = VectorDBFactory.get_index("nonexistent_alias")
            assert result is None

    def test_get_index_invalid_connection_type(self):
        """Test error handling when connection_type is invalid."""
        with patch('src.ai_factory_model.vectordb.factory.load_from_file', autospec=True) as mock_load_from_file:
            # Setup configuration with invalid connection_type
            mock_load_from_file.return_value = {
                "test_alias": {
                    "connection_type": "InvalidVectorDB",
                    "api_endpoint": "https://test.example.com",
                    "index_name": "test_index"
                }
            }

            # Call get_index and expect exception
            with pytest.raises(Exception) as exc_info:
                VectorDBFactory.get_index("test_alias")

            assert "Error in VectorDBFactory.create" in str(exc_info.value)

    def test_get_index_initialization_error(self):
        """Test error handling when vector database initialization fails."""
        with patch('src.ai_factory_model.vectordb.factory.load_from_file', autospec=True) as mock_load_from_file:
            with patch.object(AISearchVectorDB, '__init__', return_value=None):
                with patch.object(AISearchVectorDB, 'initialize_vectorDB') as mock_initialize:
                    # Setup configuration
                    config = {
                        "test_alias": {
                            "connection_type": "AISearchVectorDB",
                            "api_endpoint": "https://test.search.windows.net",
                            "api_key": "test_api_key",
                            "index_name": "test_index"
                        }
                    }
                    mock_load_from_file.return_value = config

                    # Make initialization fail
                    mock_initialize.side_effect = Exception("Connection failed")

                    # Call get_index and expect exception
                    with pytest.raises(Exception) as exc_info:
                        VectorDBFactory.get_index("test_alias")

                    assert "Error in VectorDBFactory.create" in str(exc_info.value)
                    assert "Connection failed" in str(exc_info.value)

    def test_get_index_load_file_error(self):
        """Test error handling when config file loading fails."""
        with patch('src.ai_factory_model.vectordb.factory.load_from_file', autospec=True) as mock_load_from_file:
            # Make file loading fail
            mock_load_from_file.side_effect = FileNotFoundError("Config file not found")

            # Call get_index and expect the original FileNotFoundError to be raised
            with pytest.raises(FileNotFoundError) as exc_info:
                VectorDBFactory.get_index("test_alias")

            assert "Config file not found" in str(exc_info.value)

    def test_vdb_classes_mapping(self):
        """Test that VDB_CLASSES contains expected mappings."""
        expected_classes = {
            "AISearchVectorDB": AISearchVectorDB,
            "PGVectorDB": PGVectorDB
        }

        assert VectorDBFactory.VDB_CLASSES == expected_classes

    def test_cache_key_generation(self):
        """Test that cache keys are generated correctly."""
        with patch('src.ai_factory_model.vectordb.factory.load_from_file', autospec=True) as mock_load_from_file:
            with patch.object(AISearchVectorDB, '__init__', return_value=None):
                with patch.object(AISearchVectorDB, 'initialize_vectorDB'):
                    # Setup configuration
                    config = {
                        "my_vector_db": {
                            "connection_type": "AISearchVectorDB",
                            "api_endpoint": "https://test.search.windows.net",
                            "api_key": "test_api_key",
                            "index_name": "test_index"
                        }
                    }
                    mock_load_from_file.return_value = config

                    # Call get_index
                    result = VectorDBFactory.get_index("my_vector_db")

                    # Verify cache key format
                    expected_cache_key = "index_my_vector_db"
                    assert expected_cache_key in cache
                    assert cache[expected_cache_key] == result

    def test_get_index_none_vector_def(self):
        """Test behavior when vector definition is None."""
        with patch('src.ai_factory_model.vectordb.factory.load_from_file', autospec=True) as mock_load_from_file:
            # Setup configuration that returns None for the alias
            mock_load_from_file.return_value = {
                "other_alias": {
                    "connection_type": "AISearchVectorDB"
                }
            }

            # Call get_index with alias that returns None from config.get() - should return None
            result = VectorDBFactory.get_index("test_alias")
            assert result is None

    def test_get_index_missing_connection_type_in_config(self):
        """Test error handling when connection_type is missing from config."""
        with patch('src.ai_factory_model.vectordb.factory.load_from_file', autospec=True) as mock_load_from_file:
            # Setup configuration without connection_type
            config = {
                "test_alias": {
                    "api_endpoint": "https://test.search.windows.net",
                    "api_key": "test_api_key",
                    "index_name": "test_index"
                }
            }
            mock_load_from_file.return_value = config

            # Call get_index and expect exception
            with pytest.raises(Exception) as exc_info:
                VectorDBFactory.get_index("test_alias")

            assert "Error in VectorDBFactory.create" in str(exc_info.value)

    def test_multiple_aliases_independent_caching(self):
        """Test that different aliases are cached independently."""
        with patch('src.ai_factory_model.vectordb.factory.load_from_file', autospec=True) as mock_load_from_file:
            with patch.object(AISearchVectorDB, '__init__', return_value=None):
                with patch.object(AISearchVectorDB, 'initialize_vectorDB'):
                    with patch.object(PGVectorDB, '__init__', return_value=None):
                        with patch.object(PGVectorDB, 'initialize_vectorDB'):
                            # Setup configuration for multiple aliases
                            config = {
                                "search_db": {
                                    "connection_type": "AISearchVectorDB",
                                    "api_endpoint": "https://test.search.windows.net",
                                    "api_key": "test_api_key",
                                    "index_name": "search_index"
                                },
                                "pg_db": {
                                    "connection_type": "PGVectorDB",
                                    "api_endpoint": "postgresql://user:pass@localhost/db",
                                    "index_name": "pg_table"
                                }
                            }
                            mock_load_from_file.return_value = config

                            # Get both vector databases
                            search_vdb = VectorDBFactory.get_index("search_db")
                            pg_vdb = VectorDBFactory.get_index("pg_db")

                            # Verify they are different instances and both cached
                            assert search_vdb != pg_vdb
                            assert "index_search_db" in cache
                            assert "index_pg_db" in cache
                            assert cache["index_search_db"] == search_vdb
                            assert cache["index_pg_db"] == pg_vdb

    def test_get_index_model_class_none(self):
        """Test error handling when model_class is None due to invalid connection_type."""
        with patch('src.ai_factory_model.vectordb.factory.load_from_file', autospec=True) as mock_load_from_file:
            # Setup configuration with connection_type not in VDB_CLASSES
            config = {
                "test_alias": {
                    "connection_type": "NonExistentVectorDB",
                    "api_endpoint": "https://test.example.com",
                    "index_name": "test_index"
                }
            }
            mock_load_from_file.return_value = config

            # Call get_index and expect exception due to model_class being None
            with pytest.raises(Exception) as exc_info:
                VectorDBFactory.get_index("test_alias")

            assert "Error in VectorDBFactory.create" in str(exc_info.value)
            # This should trigger a TypeError because None(vector_def) is not callable
