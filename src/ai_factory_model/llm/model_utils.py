import os
import json
import yaml
from jinja2 import Template

from ..logger import info

SEP_PATTERN = "--- message ---"


def load_from_file(file_path: str) -> dict:
    """
    Load a JSON or YAML file as a dictionary
    """

    info(f"Loading models from file \"{file_path}\"")
    if not os.path.exists(file_path):
        error_msg = f"File \"{file_path}\" does not exist"
        raise FileNotFoundError(error_msg)
    _, ext = os.path.splitext(file_path)
    try:
        with open(file_path, "r", encoding="utf-8") as file:
            if ext.lower() == ".json":
                return json.load(file)
            elif ext.lower() in {".yaml", ".yml"}:
                return yaml.safe_load(file)
            if ext.lower() == ".prompt":
                return file.read()
            else:
                error_msg = f"Not supported file format: \"{ext}\""
                raise ValueError(error_msg)
    except Exception as e:
        error_msg = f"Error loading file \"{file_path}\": {e}"
        raise RuntimeError(error_msg)


def read_template(path, params: dict):
    prompt = Template(load_from_file(path), trim_blocks=True, lstrip_blocks=True).render(**params)
    system = prompt[0:prompt.find(SEP_PATTERN)-1]
    input = prompt[prompt.find(SEP_PATTERN) + len(SEP_PATTERN):]
    return (system, input)
