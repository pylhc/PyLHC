"""
Constants: General
-------------------

General constants to be used in ``PyLHC``, to help with consistency.
"""
import numpy as np

BEAMS = (1, 2)

PLANES = ("X", "Y")
PLANE_TO_HV = dict(X="H", Y="V")

UNIT_TO_M = dict(km=1e3, m=1e0, mm=1e-3, um=1e-6, nm=1e-9, pm=1e-12, fm=1e-15, am=1e-18)

PROTON_MASS = 0.938272  # GeV/c^2
LHC_NOMINAL_EMITTANCE = 3.75 * 1e-6  # Design LHC

TFS_SUFFIX = ".tfs"
TIME_COLUMN = "TIME"


def get_proton_gamma(energy):
    """Returns relativistic gamma for protons."""
    return energy / PROTON_MASS  # E = gamma * m0 * c^2


def get_proton_beta(energy):
    """ Returns relativistic beta for protons """
    return np.sqrt(1 - (1 / get_proton_gamma(energy) ** 2))
