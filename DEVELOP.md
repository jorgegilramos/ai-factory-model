# ai-factory-model

[![PyPI version](https://img.shields.io/pypi/v/ai-factory-model.svg)](https://pypi.org/project/ai-factory-model/)
![Supported Python Versions](https://img.shields.io/pypi/pyversions/ai-factory-model)
![Build Status](https://github.com/jorgegilramos/ai-factory-model/workflows/Python%20package/badge.svg)
[![License](https://img.shields.io/badge/License-Apache%202.0-lightgrey.svg)](https://opensource.org/licenses/Apache-2.0)
[![Coverage Status](https://coveralls.io/repos/github/jorgegilramos/ai-factory-model/badge.svg?branch=main)](https://coveralls.io/github/jorgegilramos/ai-factory-model)
![PyPI - Downloads](https://img.shields.io/pypi/dm/ai-factory-model)

## Packaging

Build package
```shell
# Using build package
python -m build
```


Run tests
```shell
# All tests
pytest -q -rP

# Partial tests
pytest tests/test_application.py -v -rP
pytest tests/test_azure_auth_client.py -v -rP

# With coverage
coverage run -m pytest tests -v
coverage html
```


```shell
# Reinstall wheel avoiding reinstalling dependencies
pip install --no-deps --force-reinstall dist\ai_factory_model-0.0.11-py3-none-any.whl
```

```shell
# Reinstall wheel with dependencies
pip install dist\ai_factory_model-0.0.11-py3-none-any.whl --force-reinstall
```

```shell
# Install library from code
pip install -e .
# Install library with extra from code
pip install -e .[google_genai]
pip install -e .[community]
pip install -e .[ollama]
pip install -e .[cohere]
pip install -e .[pgvector]
```

Check style guide enforcement
```shell
flake8 . --count --exit-zero --max-complexity=10 --max-line-length=120 --statistics
```

Tox
```shell
# Test only one python version
tox -e py312
# Test all python versions
tox
```


## Uninstall

```shell
pip uninstall ai_factory_model
```

## Dependencies

| Library                | Version   | Optional dependency |
|------------------------|-----------|---------------------|
| openai                 | >= 2.31.0 | base                |
| azure-core             | >= 1.33.0 | base                |
| azure-identity         | >= 1.25.3 | base                |
| azure-keyvault-secrets | >= 4.10.0 | base                |
| langchain              | >= 1.2.15 | base                |
| langchain_openai       | >= 1.1.12 | base                |
| langchain_azure_ai     | >= 1.2.1  | base                |
| jinja2                 | >= 3.1.6  | base                |
| python-decouple        | == 3.8    | base                |
| pyyaml                 | >= 6.0.3  | base                |
| azure-search-documents | >= 11.6.0 | base                |
| langchain_google_genai | >= 4.2.1  | google_genai        |
| langchain_community    | >= 0.4.1  | community           |
| langchain_ollama       | >= 1.1.0  | ollama              |
| langchain-cohere       | >= 0.5.0  | cohere              |
| psycopg[binary]        | >= 3.2.6  | pgvector            |

# Develop requirements
| Library                | Version    |
|------------------------|------------|
| build                  | >= 1.2.2   |
| setuptools             | >= 78.1.0  |
| wheel                  | >= 0.45.1  |
| pytest                 | >= 8.3.5   |
| pytest-env             | >= 1.1.5   |
| coverage               | >= 7.8.0   |
| flake8                 | >= 7.2.0   |
| tox                    | >= 4.23.2  |


## Releases
**Version 0.0.11**:
   - Updated libraries versions
   - Implemented optional dependencies
**Version 0.0.7**:
   - Added render_template
**Version 0.0.6**:
   - First version