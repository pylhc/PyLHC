"""
Constants: General
-------------------

General Constants, useful in many places.

:module: constants.general
:author: jdilly

"""
import numpy as np

PLANES = ("X", "Y")
PLANE_TO_HV = dict(X="H", Y="V")


PROTON_MASS = 0.938272  # GeV/c^2
LHC_NOMINAL_EMITTANCE = 3.75 * 1e-6  # Design LHC


def get_proton_gamma(energy):
    """ Returns relativistic gamma for protons"""
    return energy / PROTON_MASS  # E = gamma * m0 * c^2


def get_proton_beta(energy):
    """ Returns relativistic beta for protons """
    return np.sqrt(1 - (1 / get_proton_gamma(energy)**2))
