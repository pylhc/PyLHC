
# <img src="https://twiki.cern.ch/twiki/pub/BEABP/Logos/OMC_logo.png" height="28"> PyLHC Tools

[![Travis (.com)](https://img.shields.io/travis/com/pylhc/PyLHC.svg?style=popout)](https://travis-ci.com/pylhc/PyLHC/)
[![Code Climate coverage](https://img.shields.io/codeclimate/coverage/pylhc/PyLHC.svg?style=popout)](https://codeclimate.com/github/pylhc/PyLHC)
[![Code Climate maintainability (percentage)](https://img.shields.io/codeclimate/maintainability-percentage/pylhc/PyLHC.svg?style=popout)](https://codeclimate.com/github/pylhc/PyLHC)
[![GitHub last commit](https://img.shields.io/github/last-commit/pylhc/PyLHC.svg?style=popout)](https://github.com/pylhc/PyLHC/)
[![GitHub release](https://img.shields.io/github/release/pylhc/PyLHC.svg?style=popout)](https://github.com/pylhc/PyLHC/)

This is the python-tool package of the optics measurements and corrections group (OMC).

If you are not part of that group, you will most likely have no use for the codes provided here,
unless you have a 9km wide accelerator at home.
Feel free to use them anyway, if you wish!

## Documentation

- Autogenerated docs via ``sphinx`` can be found on <https://pylhc.github.io/PyLHC/>.
- General documentation of the OMC-Teams software on <https://twiki.cern.ch/twiki/bin/view/BEABP/OMC>

## Getting Started

### Prerequisites

The codes use a multitude of packages as can be found in the [setup.py](setup.py) file.

Important ones are: ``numpy``, ``pandas`` and ``scipy``.

### Installing

This package is not deployed, hence you need to use the standard git-commands to get a local copy.

## Description

This package provides tools which can be useful for working with accelerators, but are not neccessary for
optics measurements analysis.

The latter tools can be found in [OMC3](https://github.com/pylhc/omc3) (Python 3.6) or [Beta-Beat.src](https://github.com/pylhc/Beta-Beat.src) (Python 2.7)

## Functionality

- *Machine Settings Info* - Prints an overview over the machine settings at a given time. ([**machine_settings_info.py**](https://github.com/pylhc/PyLHC/blob/master/pylhc/machine_settings_info.py))
- *MAD-X Job Submitter* - Allows to generate jobs based on a MAD-X template and submit them to HTCondor. ([**madx_submitter.py**](https://github.com/pylhc/PyLHC/blob/master/pylhc/madx_submitter.py))
- *BSRT Logger* and *BSRT Analysis* - Saves data coming straight from LHC BSRT FESA class and allows subsequent analysis. ([**BSRT_logger.py**](https://github.com/pylhc/PyLHC/blob/master/pylhc/BSRT_logger.py) & [**BSRT_analysis.py**](https://github.com/pylhc/PyLHC/blob/master/pylhc/BSRT_analysis.py) )

### Tests

- Pytest unit tests are run automatically after each commit via 
[Travis-CI](https://travis-ci.com/pylhc/PyLHC). 

### Maintainability

- Additional checks for code-complexity, design-rules, test-coverage, duplication on 
[CodeClimate](https://codeclimate.com/github/pylhc/PyLHC)

- Direct commits to master are forbidden.

## Authors

* **pyLHC/OMC-Team** - *Working Group* - [pyLHC](https://github.com/orgs/pylhc/teams/omc-team)

<!--
## License
This project is licensed under the  License - see the [LICENSE.md](LICENSE.md) file for details
-->
