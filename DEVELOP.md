# ai-factory-model
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
# Reinstall avoiding reinstalling dependencies
pip install --no-deps --force-reinstall dist\ai_factory_model-0.0.7-py3-none-any.whl
```

```shell
# Reinstall with dependencies
pip install dist\ai_factory_model-0.0.7-py3-none-any.whl --force-reinstall
```

Check style guide enforcement
```shell
flake8 . --count --exit-zero --max-complexity=10 --max-line-length=120 --statistics
```

Tox
```shell
# Command to test only one python version
tox -e py312
```


## Uninstall

```shell
pip uninstall ai_factory_model
```

## Dependencies

| Library                | Version |
|------------------------|---------|
| openai                 | 1.72.0  |
| azure-core             | 1.33.0  |
| azure-identity         | 1.21.0  |
| azure-keyvault-secrets | 4.9.0   |
| langchain              | 0.3.23  |
| langchain_openai       | 0.3.12  |
| langchain_azure_ai     | 0.1.2   |
| langchain_google_genai | 2.1.2   |
| langchain_community    | 0.3.21  |
| langchain_ollama       | 0.3.1   |
| jinja2                 | 3.1.6   |
| python-decouple        | 3.8     |
| pyyaml                 | 6.0.2   |
| azure-search-documents | 11.5.2  |
| psycopg[binary]        | 3.2.6   |
| langchain-cohere       | 0.4.3   |

# Develop requirements
| Library                | Version |
|------------------------|---------|
| build                  | 1.2.2   |
| setuptools             | 78.1.0  |
| wheel                  | 0.45.1  |
| pytest                 | 8.3.5   |
| pytest-env             | 1.1.5   |
| coverage               | 7.8.0   |
| flake8                 | 7.2.0   |
| tox                    | 4.23.2  |

## Releases
**Version 0.0.7**:
   - Added render_template
**Version 0.0.6**:
   - First version