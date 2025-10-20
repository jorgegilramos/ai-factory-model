import os
import json
import yaml
from jinja2 import Template

from ..logger import debug, error

SEP_PATTERN = "--- message ---"


def load_from_file(file_path: str) -> dict:
    """
    Load a JSON or YAML file as a dictionary
    """

    debug(f"Loading from file \"{file_path}\"")
    if not os.path.exists(file_path):
        error_msg = f"File \"{file_path}\" does not exist"
        error(error_msg)
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
                error(error_msg)
                raise ValueError(error_msg)
    except Exception as e:
        error_msg = f"Error loading file \"{file_path}\": {e}"
        error(error_msg)
        raise RuntimeError(error_msg)


def create_template(path: str) -> Template:
    return Template(load_from_file(path), trim_blocks=True, lstrip_blocks=True)


def read_template(
    path: str,
    params: dict,
    sep_pattern: str = SEP_PATTERN
) -> tuple[str, str]:

    prompt = create_template(path).render(**params)
    if prompt.find(sep_pattern) != -1:
        system = prompt[0:prompt.find(sep_pattern)].strip("\n")
        input = prompt[prompt.find(sep_pattern) + len(sep_pattern):].strip("\n")
    else:
        error_msg = f"Template separator not found: \"{sep_pattern}\""
        error(error_msg)
        raise ValueError(error_msg)
    return (system, input)


def render_template(
    template: Template,
    params: dict,
    sep_pattern: str = SEP_PATTERN
) -> tuple[str, str]:

    prompt = template.render(**params)
    if prompt.find(sep_pattern) != -1:
        system = prompt[0:prompt.find(sep_pattern)].strip("\n")
        input = prompt[prompt.find(sep_pattern) + len(sep_pattern):].strip("\n")
    else:
        error_msg = f"Template separator not found: \"{sep_pattern}\""
        error(error_msg)
        raise ValueError(error_msg)
    return (system, input)
