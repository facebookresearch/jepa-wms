# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

"""Simple utility functions for loading and dumping YAML files with preserved order and comments."""

from src.utils.logging import get_logger
import os
import re

from ruamel.yaml import YAML

logger = get_logger(__name__)


# Initialize a global YAML instance configured for round-trip mode
# This preserves order, comments, and formatting
_yaml = YAML()
_yaml.preserve_quotes = True
_yaml.default_flow_style = False


def expand_env_vars(value, _path=""):
    """
    Recursively expand environment variables in YAML values.
    Supports ${VAR_NAME} syntax and logs when variables are expanded.

    Args:
        value: The value to expand (can be str, dict, list, etc.)
        _path: Internal parameter for tracking the config path (for logging)

    Returns:
        The value with environment variables expanded
    """
    if isinstance(value, str):
        # Pattern to match ${VAR_NAME}
        pattern = r"\$\{([^}]+)\}"

        def replace_var(match):
            var_name = match.group(1)
            env_value = os.environ.get(var_name)
            if env_value is None:
                # Log warning if the env var is not set
                logger.warning(
                    f"Environment variable '{var_name}' not found{' at ' + _path if _path else ''}. "
                    f"Keeping original placeholder: {match.group(0)}"
                )
                return match.group(0)
            else:
                # Log successful expansion
                logger.info(
                    f"Expanded environment variable '{var_name}' to '{env_value}'" f"{' at ' + _path if _path else ''}"
                )
            return env_value

        return re.sub(pattern, replace_var, value)
    elif isinstance(value, dict):
        return {k: expand_env_vars(v, f"{_path}.{k}" if _path else k) for k, v in value.items()}
    elif isinstance(value, list):
        return [expand_env_vars(item, f"{_path}[{i}]") for i, item in enumerate(value)]
    else:
        return value


def load_yaml(file_path):
    """
    Load a YAML file preserving order and comments.
    Also expands environment variables in the format ${VAR_NAME}.

    Args:
        file_path (str): Path to the YAML file to load

    Returns:
        dict: The loaded YAML content with preserved order and expanded environment variables
    """
    with open(file_path, "r") as f:
        data = _yaml.load(f)
    return expand_env_vars(data)


def dump_yaml(data, file_path):
    """
    Dump data to a YAML file preserving order and comments.

    Args:
        data: The data to dump (typically a dict)
        file_path (str): Path to the output YAML file
    """
    with open(file_path, "w") as f:
        _yaml.dump(data, f)


def dumps_yaml(data):
    """
    Dump data to a YAML string preserving order and comments.

    Args:
        data: The data to dump (typically a dict)

    Returns:
        str: YAML formatted string
    """
    import io

    stream = io.StringIO()
    _yaml.dump(data, stream)
    return stream.getvalue()


def loads_yaml(yaml_string):
    """
    Load YAML from a string preserving order and comments.

    Args:
        yaml_string (str): YAML formatted string

    Returns:
        dict: The loaded YAML content with preserved order
    """
    import io

    stream = io.StringIO(yaml_string)
    return _yaml.load(stream)


def convert_to_dict_recursive(obj):
    type_name = type(obj).__name__
    if isinstance(obj, dict):
        return {k: convert_to_dict_recursive(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_to_dict_recursive(item) for item in obj]
    elif type_name in ["ScalarInt", "ScalarFloat", "ScalarString", "ScalarBoolean"]:
        if type_name == "ScalarInt":
            return int(obj)
        elif type_name == "ScalarFloat":
            return float(obj)
        elif type_name == "ScalarBoolean":
            return bool(obj)
        else:
            return str(obj)
    else:
        return obj
