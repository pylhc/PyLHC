"""
BPM Calibration
---------------

This script computes the calibration factors for the LHC BPMs using either a
beta from phase method or a dispersion one by comparison of beam optics
quantities calculated independent of and dependent on BPM calibration.
Namely, the default method compares beta-functions calculated from phase
advances (BPM-calibration independent) and from amplitude of betatron
oscillation (calibration dependent).
The other method compares dispersion, and its use is currently discouraged due
to worse resolution.

Arguments:

*--Required--*

- **inputdir** *(Path)*:

    Measurements path.

    flags: **['--input']**


- **ips** *(int)*:

    IPs to compute calibration factors for.

    flags: **['--ips']**

    choices: ``[1, 4, 5]``


- **outputdir** *(Path)*:

    Output directory where to write the calibration factors.

    flags: **['--outputdir']**


*--Optional--*

- **method** *(str)*:

    Method to be used to compute the calibration factors. The Beta
    function is used by default.

    flags: **['--method']**

    choices: ``('beta', 'dispersion')``

    default: ``beta``
"""
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd
import tfs
from generic_parser import EntryPointParameters, entrypoint
from omc3.optics_measurements.constants import EXT
from omc3.utils import logging_tools

from pylhc.calibration.beta import get_calibration_factors_from_beta
from pylhc.calibration.dispersion import get_calibration_factors_from_dispersion
from pylhc.constants.calibration import CALIBRATION_NAME, IPS, METHODS

LOG = logging_tools.get_logger(__name__)


def _get_params() -> dict:
    """
    Parse Commandline Arguments and return them as options.

    Returns:
        dict
    """

    return EntryPointParameters(
        inputdir=dict(
            type=Path,
            required=True, 
            help="Measurements path."
        ),
        outputdir=dict(
            type=Path,
            required=True,
            help="Output directory where to write the calibration factors.",
        ),
        ips=dict(
            type=int,
            nargs="+",
            choices=IPS,
            required=False,
            help="IPs to compute calibration factors for.",
        ),
        method=dict(
            type=str,
            required=False,
            choices=METHODS,
            default=METHODS[0],
            help=(
                "Method to be used to compute the calibration factors. "
                "The Beta function is used by default."
            ),
        ),
    )


@entrypoint(_get_params(), strict=True)
def main(opt):
    # Compute the calibration factors and their errors according to the method
    if opt.method == "beta":
        factors = get_calibration_factors_from_beta(opt.ips, opt.inputdir)
    elif opt.method == "dispersion":
        factors = get_calibration_factors_from_dispersion(opt.ips, opt.inputdir)

    # Fill NaN with 1 because of missing BPMs and that fit cannot be done everywhere
    for plane in factors.keys():
        factors[plane] = factors[plane].fillna(1)
    LOG.debug("".join([f"\nPlane {plane}:\n{factors[plane]}" for plane in factors.keys()]))

    # Write the TFS file to the desired output directory
    opt.outputdir.mkdir(parents=True, exist_ok=True)
    for plane in factors.keys():
        tfs.write(opt.outputdir / f"{CALIBRATION_NAME[opt.method]}{plane.lower()}{EXT}", 
                  factors[plane].reset_index(), 
                  save_index=False)

    return factors


if __name__ == "__main__":
    main()
