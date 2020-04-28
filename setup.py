import re
import os
import pathlib
from setuptools import setup, find_packages

# The directory containing this file
HERE = pathlib.Path(__file__).parent

# The text of the README file
README = (HERE / "README.md").read_text()

# Name of the module
MODULE_NAME = 'pylhc'

# Dependencies for the module itself
DEPENDENCIES = [
    'numpy>=1.17.1',
    'scipy>=1.0.0',
    'pandas>=0.24.0<1.0',
    'GitPython>=2.1.8',
    'matplotlib>=3.1.0',
    'ruamel.yaml>=0.15.94',
    'cmmnbuild-dep-manager>=2.2.2<=2.3.0',
    'pjlsa>=0.0.14',
    'pytimber>=2.6.2',
    'htcondor>=8.9.2',
    'tfs-pandas>=1.0.3',
    'generic-parser>=1.0.6',
    'JPype1>=0.7.2',
    'parse>=1.15.0'
    'ipython>=7.0.1',  # actually dependency of pytimber
    'omc3@https://github.com/pylhc/omc3/tarball/master',  # installed in travis.yml
    # 'pyjapc@https://gitlab.cern.ch/scripting-tools/pyjapc/tarball/master'
]

# Test dependencies that should only be installed for test purposes
TEST_DEPENDENCIES = ['pytest>=5.2',
                     'pytest-cov>=2.6',
                     'pytest-mpl>=0.11',
                     'hypothesis>=4.36.2',
                     'attrs>=19.2.0'
                     ]

# pytest-runner to be able to run pytest via setuptools
SETUP_REQUIRES = ['pytest-runner']

# Extra dependencies for tools
EXTRA_DEPENDENCIES = {'doc': ['sphinx',
                              'travis-sphinx',
                              'sphinx_rtd_theme']
                      }


def get_version():
    """ Reads package version number from package's __init__.py. """
    with open(os.path.join(
            os.path.dirname(__file__), MODULE_NAME, '__init__.py'
    )) as init:
        for line in init.readlines():
            res = re.match(r'^__version__ = [\'"](.*)[\'"]$', line)
            if res:
                return res.group(1)


# This call to setup() does all the work
setup(
    name=MODULE_NAME,
    version=get_version(),
    description="Useful tools in particular for accelerator physicists at CERN",
    long_description=README,
    long_description_content_type="text/markdown",
    url="https://github.com/pylhc/pylhc",
    author="pyLHC",
    author_email="pylhc@github.com",
    license="MIT",
    classifiers=[
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.6",
    ],
    packages=find_packages(include=(MODULE_NAME,)),
    install_requires=DEPENDENCIES,
    tests_require=DEPENDENCIES + TEST_DEPENDENCIES,
    extras_require=EXTRA_DEPENDENCIES,
    setup_requires=SETUP_REQUIRES,
)