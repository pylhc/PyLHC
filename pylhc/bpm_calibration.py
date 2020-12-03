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
import numpy as np
import pandas as pd
from typing import Tuple, List, Dict

from generic_parser import EntryPointParameters, entrypoint
from omc3.utils import logging_tools
from omc3.optics_measurements.constants import EXT
from pylhc.constants.calibration import (
    CALIBRATION_NAME,
    IPS,
    METHODS,
)
from pylhc.calibration.dispersion import get_calibration_factors_from_dispersion
from pylhc.calibration.beta import get_calibration_factors_from_beta
import tfs


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


def _write_calibration_tfs(
    calibration_factors: pd.DataFrame, plane: str, method: str, outputdir: Path
) -> None:
    """
    This function saves to a file the calibration factors given as input.
    The file name contains the method and plane related to those factors.
    e.g: "calibration_beta_x.tfs"

    Args:
      calibration_factors (pd.DataFrame): The DataFrame containing the
        calibration factors to be written to disk.
      plane (str): The plane associated to the current calibration factors.
      method (str): The method used to compute those calibration factors.
      outputdir (Path): The directory where to save thedir.

    Returns:
      None
    """
    # Create the output directory
    outputdir.mkdir(parents=True, exist_ok=True)

    # Reset the index of the dataframe, it was handy to handle the data but not
    # to store it
    calibration_factors = calibration_factors.reset_index()

    # Write the TFS files for this plane
    # The method chosen will change the tfs name
    tfs_name = f"{CALIBRATION_NAME[method]}{plane.lower()}{EXT}"
    file_path = outputdir / tfs_name
    LOG.info(f"Writing {file_path}")
    tfs.write_tfs(file_path, calibration_factors, save_index=False)


def _get_str_calibration_factors(calibration_factors: Dict[str, pd.DataFrame]) -> str:
    """
    Return the calibration factor in a console readable format

    Args:
        calibration_factors (Dict[str, pd.DataFrame]): the dataframe containing
        all the calibration factors for each plane

    Returns:
        str: The calibration factors as string
    """
    result_str = ""
    for plane in calibration_factors.keys():
        result_str += f"\nPlane {plane}:\n"
        result_str += f"{calibration_factors[plane]}"

    return result_str


@entrypoint(_get_params(), strict=True)
def main(opt):
    # Compute the calibration factors and their errors according to the method
    if opt.method == "beta":
        factors = get_calibration_factors_from_beta(opt.ips, opt.inputdir)
    elif opt.method == "dispersion":
        factors = get_calibration_factors_from_dispersion(opt.ips, opt.inputdir)

    # Fill NaN with 1 because of missing BPMs and that fit cannot be done everywhere
    for plane in factors.keys():
        factors[plane].fillna(1, inplace=True)
    LOG.debug(_get_str_calibration_factors(factors))

    # Write the TFS file to the desired output directory
    for plane in factors.keys():
        _write_calibration_tfs(factors[plane], plane, opt.method, opt.outputdir)

    return factors


if __name__ == "__main__":
    main()
