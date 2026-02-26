# -----------------------------------------------------------------------------.
# MIT License

# Copyright (c) 2025 RadDB developers
#
# This file is part of RadDB.

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

# -----------------------------------------------------------------------------.
"""RadDB configurations settings."""
import os

from raddb.utils.yaml import read_yaml, write_yaml


def _define_config_filepath():
    """Define the config YAML file path."""
    # Retrieve user home directory
    home_directory = os.path.expanduser("~")
    # Define path where .config_raddb.yaml file should be located
    return os.path.join(home_directory, ".config_raddb.yaml")


def define_configs(
    base_dir: str | None = None,
):
    """Defines the RadDB configuration file with the given credentials and base directory.

    Parameters
    ----------
    base_dir : str
        The base directory where radar data are stored.

    Notes
    -----
    This function writes a YAML file to the user's home directory at ~/.config_raddb.yaml
    with the given RadDB credentials and base directory. The configuration file can be
    used for authentication when making RadDB requests.

    """
    # Define path to .config_raddb.yaml file
    filepath = _define_config_filepath()

    # If the config exists, read it and update it ;)
    if os.path.exists(filepath):
        config_dict = read_yaml(filepath)
        action_msg = "updated"
    else:
        config_dict = {}
        action_msg = "written"

    # Add RadDB base directory
    if base_dir is not None:
        config_dict["base_dir"] = str(base_dir)  # deal with Pathlib

    # Write the RadDB config file
    write_yaml(config_dict, filepath, sort_keys=False)

    print(f"The RadDB config file has been {action_msg} successfully!")


def read_configs() -> dict[str, str]:
    """Reads the RadDB configuration file and returns a dictionary with the configuration settings.

    Returns
    -------
    dict
        A dictionary containing the configuration settings for the RadDB.

    Raises
    ------
    ValueError
        If the configuration file has not been defined yet. Use `raddb.define_configs()` to
        specify the configuration file path and settings.

    Notes
    -----
    This function reads the YAML configuration file located at ~/.config_raddb.yaml, which
    should contain the RadDB credentials and base directory specified by `raddb.define_configs()`.

    """
    # Define path to .config_raddb.yaml file
    filepath = _define_config_filepath()
    # Check it exists
    if not os.path.exists(filepath):
        raise ValueError(
            "The RadDB config file has not been specified. Use raddb.define_configs to specify it !",
        )
    # Read the RadDB config file
    return read_yaml(filepath)


####--------------------------------------------------------------------------.
def _get_config_key(key):
    """Return the config key."""
    import raddb

    value = raddb.config.get(key, None)
    if value is None:
        raise ValueError(f"The '{key}' is not specified in the RadDB configuration file.")
    return value


def get_base_dir(base_dir=None):
    """Return the RadDB base directory."""
    import raddb

    if base_dir is None:
        base_dir = raddb.config.get("base_dir")
    if base_dir is None:
        raise ValueError("The 'base_dir' is not specified in the RadDB configuration file.")
    return str(base_dir)  # convert Path to str
