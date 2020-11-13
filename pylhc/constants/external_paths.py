"""
Constants: External Paths
-------------------------

Specific constants relating to external paths to be used in ``PyLHC``, to help with consistency.
"""
import os

# Binary Files -----------------------------------------------------------------
MADX_BIN = os.path.join("/", "afs", "cern.ch", "user", "m", "mad", "bin", "madx")
PYTHON3_BIN = os.path.join(
    "/", "afs", "cern.ch", "eng", "sl", "lintrack", "anaconda3", "bin", "python"
)
PYTHON2_BIN = os.path.join(
    "/", "afs", "cern.ch", "eng", "sl", "lintrack", "miniconda2", "bin", "python"
)
