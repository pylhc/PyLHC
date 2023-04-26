# pylhc Changelog

## Version 0.8.0

Fixed:
 - replaced `get_loc` with `get_indexer` to be compatible with `pandas` > 2.0

Added:
 - `beamprocess` option to Machine Settings Info, to allow manual override.

Removed:
 - `hdf5` reader/writer for `tfs` as this is already in `tfs-pandas`

## Version 0.7.4

Changes in Machine Settings Info
 - Default behaviour for no knobs given: extract None.
 - Old behaviour of extracting all restored by giving `knobs = ["all"]`
 - Option `["default"]` available for default knobs.
 - Additional debug logging

## Version 0.7.3 

Fixes:
 - LSA to MADX sign convention
 - Better trim naming
 - check madx names for allowed characters
 - option to init all variables
 - machine settings info takes ISO time

## Version 0.7.2

Fix: Trim variables generated in the MAD-X script will make sure not to be longer than 47 characters (hard MAD-X limit), nor start with an underscore or a digit.

## Version 0.7.1

Fix: Will not attempt to write to disk knobs that were not found by LSA in the provided optics.

## Version 0.7.0

Added `LSA` knobs to `MAD-X` conversion capability: given LSA knobs and the optics they apply to, creates `.madx` files that will reproduce the knobs in simulation.

## Version 0.6.2

Added a flag to the `info` functionality of `pylhc.kickgroups` to display a copy-pastable list of kick files for the GUI to load at once.

## Version 0.6.1

Bugfixes in KickGroups:
  - Better error message when there are no kicks in group.
  - Find correct planes in lists of data.
  - Renamed functions to `list` and `info`.

## Version 0.6.0

Added KickGroups functionality: load available kick groups in a folder, display their relevant information; and do the same for all kick files in a given kickgroup.

## Version 0.5.0

Removed `irnl_rdt_correction`. Is now in https://github.com/pylhc/irnl_rdt_correction

## Version 0.4.1

Minor bugfixes in `machine_settings_info`.

- Added:
  - `time` and `start_time` can now be given as `AccDatetime`-objects.
  
- Fixed:
  - `trims` variable is initialized as `None`. Was not initialized if no 
  trims were found, but used later on.
    

## Version 0.4.0

* Add Zenodo DOI to README by @fsoubelet in https://github.com/pylhc/PyLHC/pull/89
* Adds check for non-existing knobs by @JoschD in https://github.com/pylhc/PyLHC/pull/90
* Update CI by @fsoubelet in https://github.com/pylhc/PyLHC/pull/91
* Lsa with timerange by @JoschD in https://github.com/pylhc/PyLHC/pull/92

Release `0.4.0` brings the trim-history option to the machine-info extractor.
To enable this, one needs to provide a `start_time`.
The return values are now organized into a dictionary.

**Full Changelog**: https://github.com/pylhc/PyLHC/compare/0.3.0...v0.4.0


## Version 0.3.0

Release `0.3.0` brings the following:

Added:
- Non-linear correction script for the (HL)LHC Insertion Regions Resonance Driving Terms, including feed-down effects.

Changed:
- The package's license has been moved from `GPLv3` to `MIT`.

Note: if one wishes to extend the `IRNL` correction script to a different accelerator, 
there are valuable pointers in the following 
[PR comment](https://github.com/pylhc/PyLHC/pull/74#issuecomment-966212021).


## Version 0.2.0

This is the first release of `pylhc` since its `omc3` dependency is available on `PyPI`.

Added:
- BPM calibration script to get calibration factors from different BPMs
- Proper mocking of CERN TN packages (functionality imported from `omc3`)

Changed:
- Minimum required `tfs-pandas` version is now `3.0.2`
- Minimum required `generic-parser` version is now `1.0.8`
- Minimum required `omc3` version is now `0.2.0`
- Extras related to the CERN TN are now installed with `python -m pip install pylhc[cern]`

Removed:
- The `HTCondor` and `AutoSix` functionality have been removed and extracted to another package at https://github.com/pylhc/submitter


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
