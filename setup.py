import pathlib

import setuptools

# The directory containing this file
MODULE_NAME = "pylhc"
TOPLEVEL_DIR = pathlib.Path(__file__).parent.absolute()
ABOUT_FILE = TOPLEVEL_DIR / MODULE_NAME / "__init__.py"
README = TOPLEVEL_DIR / "README.md"


def about_package(init_posixpath: pathlib.Path) -> dict:
    """
    Return package information defined with dunders in __init__.py as a dictionary, when
    provided with a PosixPath to the __init__.py file.
    """
    about_text: str = init_posixpath.read_text()
    return {
        entry.split(" = ")[0]: entry.split(" = ")[1].strip('"')
        for entry in about_text.strip().split("\n")
        if entry.startswith("__")
    }


ABOUT_PYLHC = about_package(ABOUT_FILE)

with README.open("r") as docs:
    long_description = docs.read()

# Dependencies for the module itself
DEPENDENCIES = [
    "numpy>=1.19",
    "scipy>=1.4.0",
    "pandas>=1.0,<1.2",  # limit because of https://github.com/pandas-dev/pandas/issues/39872
    "matplotlib>=3.2.0",
    "pjlsa>=0.0.14",
    "pytimber>=2.8.0",
    "tfs-pandas>=2.0",
    "generic-parser>=1.0.8",
    "parse>=1.15.0",
    "omc3@https://github.com/pylhc/omc3/tarball/master",
]

EXTRA_DEPENDENCIES = {
    "tech": [
        "jpype1<0.8.0,>=0.7.3",  # limit from pylsa
        # "cmmnbuild-dep-manager/@https://gitlab.cern.ch/scripting-tools/cmmnbuild-dep-manager/repository/archive.tar.gz?ref=master",
        "pyjapc@https://gitlab.cern.ch/scripting-tools/pyjapc/repository/archive.tar.gz?ref=master",
    ],
    "test": [
        "pytest>=5.2",
        "pytest-cov>=2.7",
        "pytest-regressions>=2.0.0",
        "pytest-mpl>=0.11",
    ],
    "doc": ["sphinx", "sphinx_rtd_theme"],
}
EXTRA_DEPENDENCIES.update(
    {"all": [elem for list_ in EXTRA_DEPENDENCIES.values() for elem in list_]}
)


setuptools.setup(
    name=ABOUT_PYLHC["__title__"],
    version=ABOUT_PYLHC["__version__"],
    description=ABOUT_PYLHC["__description__"],
    long_description=long_description,
    long_description_content_type="text/markdown",
    author=ABOUT_PYLHC["__author__"],
    author_email=ABOUT_PYLHC["__author_email__"],
    url=ABOUT_PYLHC["__url__"],
    python_requires=">=3.7",
    license=ABOUT_PYLHC["__license__"],
    classifiers=[
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Natural Language :: English",
        "Programming Language :: Python",
        "Programming Language :: Python :: 3 :: Only",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Topic :: Scientific/Engineering :: Physics",
        "Topic :: Scientific/Engineering :: Visualization",
    ],
    packages=setuptools.find_packages(exclude=["tests*", "doc"]),
    include_package_data=True,
    install_requires=DEPENDENCIES,
    tests_require=EXTRA_DEPENDENCIES["test"],
    extras_require=EXTRA_DEPENDENCIES,
)
