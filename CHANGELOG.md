# pylhc Changelog

## Version 0.1.1

- Added:
    - `python2` parameter for autosix.
  
- Changed:
    - Bugfix for non-unique column names when indexing (`forced_da_analysis`)

## Version 0.1.0

- Added:
    - Job submitter script to easily generate and schedule jobs through HTCondor.
    - Autosix script to easily generate and submit parametric SixDesk studies through HTCondor.
    - Script to analyse forced dynamic aperture data.
    - Scripts for logging and analysis of LHC BSRT data.
    - Utility modules supporting functionality for the above scripts.

- Changed:
    - License moved to GNU GPLv3 to comply with the use of the `omc3` package.

- Miscellaneous:
    - Introduced extra dependencies tailored to different use cases of the package.
    - Reworked package organisation for consistency.
    - Set minimum requirements versions.
    - Moved CI/CD setup to Github Actions.
    - Improved testing and test coverage.

## Version 0.0.2

No changes somehow.

## Version 0.0.1

- Added:
    - Script to retrieve machine settings information.

- Removed:
    - All previous outdated files.
