"""
Constants: BPM Resynchronization
--------------------------------

Specific constants related to the BPM resynchronization script in ``PyLHC``.
"""

from typing import Final, Literal

# Available rings
RINGS: Final[set[Literal['LER', 'HER']]] = {'LER', 'HER'}

# Phase file containing the phase advance of the BPMs
PHASE_FILE: Final[str] = "total_phase_{plane}.tfs"

DEFAULT_DATATYPE: Final[Literal['lhc']] = "lhc"
