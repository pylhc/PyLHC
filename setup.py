import re
import sys
import os
import pathlib
import shlex
from setuptools import setup, find_packages

from setuptools.command.test import test as TestCommand


class PyTest(TestCommand):
    """ Allows passing commandline arguments to pytest.
        e.g. `python setup.py test -a='-o python_classes=BasicTests'`
        or   `python setup.py pytest -a '-o python_classes="BasicTests ExtendedTests"'
        or   `python setup.py test --pytest-args='--collect-only'`
    """
    user_options = [('pytest-args=', 'a', "Arguments to pass into pytest")]

    def initialize_options(self):
        TestCommand.initialize_options(self)
        self.pytest_args = ''

    def finalize_options(self):
        TestCommand.finalize_options(self)
        self.test_args = []
        self.test_suite = True

    def run_tests(self):
        import pytest

        # shlex.split() preserves quotes
        errno = pytest.main(shlex.split(self.pytest_args))
        sys.exit(errno)


# The directory containing this file
HERE = pathlib.Path(__file__).parent

# The text of the README file
README = (HERE / "README.md").read_text()

# Name of the module
MODULE_NAME = 'pylhc'

# Dependencies for the module itself
DEPENDENCIES = [
    'numpy>=1.18.0',
    'scipy>=1.4.0',
    'pandas==0.25.*',
    'jpype1<0.8.0,>=0.7.3',  # limit from pylsa
    'ipython>=7.0.1',  # actually dependency of pytimber
    'GitPython>=2.1.8',
    'matplotlib>=3.2.0',
    'ruamel.yaml>=0.15.94',
    'cmmnbuild-dep-manager>=2.2.2<=2.3.0',
    'pjlsa>=0.0.14',
    'pytimber>=2.8.0',
    'htcondor>=8.9.2',
    'tfs-pandas>=1.0.3',
    'generic-parser>=1.0.6',
    'parse>=1.15.0',
    'omc3@https://github.com/pylhc/omc3/tarball/master',  # to be installed by user (see travis.yml)
    'pyjapc@https://gitlab.cern.ch/scripting-tools/pyjapc/tarball/master'  # to be installed by user (see travis.yml)
]


EXTRA_DEPENDENCIES = {
    "setup": [
        "pytest-runner"
    ],
    "test": [
        "pytest>=5.2",
        "pytest-cov>=2.7",
        'pytest-regressions>=2.0.0',
        'pytest-mpl>=0.11',
        "hypothesis>=5.0.0",
        "attrs>=19.2.0",
    ],
    "doc": [
        "sphinx",
        "travis-sphinx",
        "sphinx_rtd_theme"
    ],
}
EXTRA_DEPENDENCIES.update(
    {'all': [elem for list_ in EXTRA_DEPENDENCIES.values() for elem in list_]}
)


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
    python_requires=">=3.6",
    license="MIT",
    cmdclass={'pytest': PyTest},  # pass test arguments
    classifiers=[
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.6",
    ],
    packages=find_packages(exclude=['tests*', 'doc']),
    install_requires=DEPENDENCIES,
    tests_require=EXTRA_DEPENDENCIES['test'],
    setup_requires=EXTRA_DEPENDENCIES['setup'],
    extras_require=EXTRA_DEPENDENCIES,
)
