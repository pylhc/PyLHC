"""
Constants: External Paths
----------------------------------

Collections of external paths used in this package.

:module: constants.forced_da_analysis
:author: jdilly

"""
from pathlib import Path

# Binary Files -----------------------------------------------------------------
MADX_BIN = Path('/', 'afs', 'cern.ch', 'user', 'm', 'mad', 'bin', 'madx')
PYTHON3_BIN = Path('/', 'afs', 'cern.ch', 'eng', 'sl', 'lintrack', 'anaconda3', 'bin', 'python')
PYTHON2_BIN = Path('/', 'afs', 'cern.ch', 'eng', 'sl', 'lintrack', 'miniconda2', 'bin', 'python')
SIXDESK_UTILS = Path('/afs/cern.ch/project/sixtrack/SixDesk_utilities/pro/utilities/bash/')


# Repositories -----------------------------------------------------------------
SIXDESK_GITHUB = "https://github.com/SixTrack/SixDesk.git"
