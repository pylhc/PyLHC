import pathlib
from setuptools import setup

# The directory containing this file
HERE = pathlib.Path(__file__).parent

# The text of the README file
README = (HERE / "README.md").read_text()

# This call to setup() does all the work
setup(
    name="pylhc",
    version="0.0.2",
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
    packages=["pylhc"],
    install_requires=["numpy", "pandas"],
)