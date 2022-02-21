"""
Dispersion
----------

The functions in this script compute the calibration factors for the LHC BPMs
using the dispersion method. The `get_calibration_factors_from_dispersion` is
intended to be used with the script `bpm_calibration.py`.

"""
from pathlib import Path
from scipy.optimize import curve_fit
import numpy as np
import pandas as pd
from typing import Tuple, List, Dict

from omc3.utils import logging_tools
from omc3.optics_measurements.constants import (
    ERR,
    EXT,
    NORM_DISP_NAME,
    DISPERSION_NAME,
    S,
)

from pylhc.constants.calibration import (
    BPMS,
    D_BPMS,
    D,
    IPS,
    LABELS,
    ND,
    TFS_INDEX,
)
from pylhc.constants.general import PLANES
import tfs


LOG = logging_tools.get_logger(__name__)


def _get_dispersion_fit(
    positions: pd.Series, dispersion_values: pd.Series, dispersion_err: pd.Series
) -> Tuple[pd.Series, pd.Series]:
    """
    This function returns a fit of the given dispersion values along with the
    associated error.

    Args:
      positions (pd.Series): Positions of the BPMs to be fitted.
      dispersion_values (pd.Series): Values of the BPMs to be fitted.
      dispersion_err (pd.Series): Associated errors to the values.

    Returns:
      Tuple[pd.Series, pd.Series]: The elements returned are the values of the
      fit of the dispersion values and the associated error.
    """

    def dispersion_function(x, a, b):
        return a * x + b

    # Filter the values we have to only keep the asked BPMs
    values = dispersion_values[dispersion_values.index.isin(positions.index)]
    err = dispersion_err[dispersion_values.index.isin(positions.index)]

    # Get the curve fit for the expected affine function
    valid = ~(np.isnan(positions) | np.isnan(values))
    fit, fit_cov = curve_fit(dispersion_function, positions[valid], values[valid], sigma=err[valid])

    # Get the error from the covariance matrix
    fit_err = np.sqrt(np.diag(fit_cov))

    # Get the fitted beta and add the errors to get min/max values
    dispersion_fit = dispersion_function(positions, fit[0], fit[1])
    dispersion_max_fit = dispersion_function(
        positions, fit[0] + fit_err[0], fit[1] + fit_err[1]
    )
    dispersion_min_fit = dispersion_function(
        positions, fit[0] - fit_err[0], fit[1] - fit_err[1]
    )
    dispersion_fit_err = (dispersion_max_fit - dispersion_min_fit) / 2

    return dispersion_fit, dispersion_fit_err


def _get_factors_from_dispersion(
        dispersion: Dict[str, pd.Series], phase: str, phase_err: str,
) -> Tuple[pd.Series, pd.Series]:
    """
    This function computes the calibration factors for the dispersion method
    with the non fitted dispersion values. The associated error is also
    calculated.
    The equations being the same for the factors from dispersion and dispersion
    fit, this function can be used for both.

    Args:
      dispersion (Dict[str, pd.Series]): Dictionnary containing 4 keys: phase,
        phase_err, amp and amp_err. Each key is related to the method used to
        obtain the dispersion and its error. The keys differ for the dispersion
        fit method, as a _fit part is added.
      phase (str): key for the dispersion from phase
      phase_err (str): key for the dispersion from phase error

    Returns:
      Tuple[pd.Series, pd.Series]: The first Series are the calibration
        factors, the second one their error.
    """
    # Get the ratios, those are our calibration factors
    factors = dispersion[phase] / dispersion["amp"]

    # Code in BBs
    calibration_error = (dispersion[phase_err] / dispersion["amp"]) ** 2
    calibration_error += (
        dispersion["amp_err"] * (dispersion[phase] / (dispersion["amp"] ** 2))
    ) ** 2
    calibration_error = np.sqrt(calibration_error)

    return factors, calibration_error


def get_calibration_factors_from_dispersion(
    ips: List[int], input_path: Path
) -> Dict[str, pd.DataFrame]:
    """
    This function is the main function to compute the calibration factors for
    the dispersion method.
    Given an IP and a path containing the corresponding Tfs files, this
    function returns the calibration factors using both the dispersion and its
    fitted values.
    The calibration factors based on the dispersion are only computed for the
    X plane.

    Args:
      ips (List[int]): IPs to compute the calibration factors for.
      input_path (Path): Path of the directory containing the beta files.

    Returns:
      Dict[str, pd.DataFrame]: The returned DataFrame object contains the
      calibration factors for each BPM along with their error. Both the
      dispersion and dispersion from fit values are used, resulting in 6
      columns:

        - NAME: BPM Name
        - S: Position
        - CALIBRATION: Calibration factors computed from the dispersion
        - ERROR_CALIBRATION: Associated error to the above calibration
          factors
        - CALIBRATION_FIT: Calibration factors computed from the fitted
          dispersion
        - ERROR_CALIBRATION_FIT: Associated error to the above calibration
          factors
    """
    LOG.info("Computing the calibration factors via dispersion")
    # Load the normalized dispersion tfs file
    norm_dispersion_tfs = tfs.read(
        input_path / f"{NORM_DISP_NAME}x{EXT}", index=TFS_INDEX
    )
    dispersion_tfs = tfs.read(input_path / f"{DISPERSION_NAME}x{EXT}", index=TFS_INDEX)

    # Get the beam concerned by those tfs files
    beam = int(dispersion_tfs.iloc[0].name[-1])

    # Loop over the IPs and compute the calibration factors
    calibration_factors = dict()
    for ip in ips:
        LOG.info(f"  Computing the calibration factors for IP {ip}, plane X")
        # Filter our TFS files to only keep the BPMs for the selected IR
        bpms = dispersion_tfs.reindex(BPMS[ip][beam])
        d_bpms = dispersion_tfs.reindex(D_BPMS[ip][beam])
            
        # Check for possible missing bpms
        for bpm_set in [bpms, d_bpms]:
            missing = set(bpm_set.loc[bpm_set.isnull().values].index)
            if missing:
                LOG.warning("    One or several BPMs are missing in the input"
                            " DataFrame, the calibration factors calculation"
                            f"from fit may not be accurate: {missing}")

        # Get the positions of the BPMs and the subset used for the fit
        bpms = bpms.index
        d_bpms = d_bpms.index
        positions = dispersion_tfs.reindex(bpms)[S]
        positions_fit = dispersion_tfs.reindex(d_bpms)[S]

        # Get the dispersion and dispersion from phase from the tfs files
        dispersion = dict()
        normalised_dispersion = dict()

        dispersion["amp"] = dispersion_tfs.reindex(bpms)[f"DX"]
        dispersion["amp_err"] = dispersion_tfs.reindex(bpms)[f"{ERR}{D}X"]

        dispersion["phase"] = norm_dispersion_tfs.reindex(bpms)[f"DX"]
        dispersion["phase_err"] = norm_dispersion_tfs.reindex(bpms)[f"{ERR}{D}X"]

        # Compute the calibration factors using the dispersion from phase and amp
        calibration, calibration_err = _get_factors_from_dispersion(
            dispersion, "phase", "phase_err")

        # Fit the dispersion from phase
        dispersion["phase_fit"], dispersion["phase_fit_err"] = _get_dispersion_fit(
            positions_fit, dispersion["phase"], dispersion["phase_err"]
        )

        # Compute the calibration factors using the fitted dispersion from amp / phase
        calibration_fit, calibration_fit_err = _get_factors_from_dispersion(
            dispersion, "phase_fit", "phase_fit_err"
        )

        # Assemble the calibration factors in one dataframe
        factors_for_ip = pd.concat(
            [
                positions,
                calibration,
                calibration_err,
                calibration_fit,
                calibration_fit_err,
            ],
            axis=1,
        )
        factors_for_ip.columns = LABELS
        factors_for_ip.index.name = TFS_INDEX

        if "X" not in calibration_factors.keys():
            calibration_factors = {"X": factors_for_ip}
        else:
            calibration_factors["X"] = calibration_factors["X"].append(factors_for_ip)

    return calibration_factors
