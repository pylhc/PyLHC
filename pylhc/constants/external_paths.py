"""
Constants: External Paths
-------------------------

Specific constants relating to external paths to be used in ``PyLHC``,
to help with consistency.
"""
from pathlib import Path

AFS_CERN = Path('/', 'afs', 'cern.ch')
LINTRACK = AFS_CERN / 'eng' / 'sl' / 'lintrack'

# Binary Files -----------------------------------------------------------------
MADX_BIN = AFS_CERN / 'user' / 'm' / 'mad' / 'bin' / 'madx'
PYTHON3_BIN = LINTRACK / 'anaconda3' / 'bin' / 'python'
PYTHON2_BIN = LINTRACK / 'miniconda2' / 'bin' / 'python'
SIXDESK_UTILS = AFS_CERN / 'project' / 'sixtrack' / 'SixDesk_utilitiespro' / 'utilities' / 'bash'


# Repositories -----------------------------------------------------------------
SIXDESK_GITHUB = "https://github.com/SixTrack/SixDesk.git"
