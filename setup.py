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
TOPLEVEL_DIR = pathlib.Path(__file__).parent.absolute()
ABOUT_FILE = TOPLEVEL_DIR / "pylhc" / "__init__.py"
README = TOPLEVEL_DIR / "README.md"

# Information on the omc3 package
ABOUT: dict = {}
with ABOUT_FILE.open("r") as f:
    exec(f.read(), ABOUT)

with README.open("r") as docs:
    long_description = docs.read()


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
    # to be installed by user (see travis.yml and README):
    'omc3@https://github.com/pylhc/omc3/tarball/master',
    'pyjapc@https://gitlab.cern.ch/scripting-tools/pyjapc/repository/archive.tar.gz?ref=master'
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


# This call to setup() does all the work
setup(
    name=ABOUT["__title__"],
    version=ABOUT["__version__"],
    description=ABOUT["__description__"],
    long_description=long_description,
    long_description_content_type="text/markdown",
    author=ABOUT["__author__"],
    author_email=ABOUT["__author_email__"],
    url=ABOUT["__url__"],
    python_requires=">=3.6",
    license=ABOUT["__license__"],
    cmdclass={'pytest': PyTest},  # pass test arguments
    classifiers=[
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Natural Language :: English",
        "Programming Language :: Python",
        "Programming Language :: Python :: 3 :: Only",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Topic :: Scientific/Engineering :: Physics",
        "Topic :: Scientific/Engineering :: Visualization",
    ],
    packages=find_packages(exclude=['tests*', 'doc']),
    install_requires=DEPENDENCIES,
    tests_require=EXTRA_DEPENDENCIES['test'],
    setup_requires=EXTRA_DEPENDENCIES['setup'],
    extras_require=EXTRA_DEPENDENCIES,
)
