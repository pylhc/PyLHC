"""
Logging Tools
--------------

Super simple logging tools.
Might be replaced in the future with logging_tools form omc3.

:module: utils.logging_tools
:author: jdilly

"""
import logging.config
import os

from ruamel import yaml

LOGGING_CONFIG = os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir, 'logging_config.yml')


def setup_logger():
    """ Setup logging. """
    with open(LOGGING_CONFIG) as stream:
        logging.config.dictConfig(yaml.safe_load(stream))


