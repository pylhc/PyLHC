"""
beta
-----

The functions in this script compute the calibration factors for the LHC BPMs
using the beta method. The `get_calibration_factors_from_beta` is intended
to be used with the script `bpm_calibration.py`.

"""
from pathlib import Path
from scipy.optimize import curve_fit
import numpy as np
import pandas as pd
from typing import Tuple, List, Dict

from omc3.utils import logging_tools
from omc3.optics_measurements.constants import (
    AMP_BETA_NAME,
    BETA,
    BETA_NAME,
    ERR,
    EXT,
    S,
)
from pylhc.constants.calibration import (
    BPMS,
    BETA_STAR_ESTIMATION,
    IPS,
    LABELS,
    TFS_INDEX,
)
from pylhc.constants.general import PLANES
import tfs


LOG = logging_tools.get_logger(__name__)


def _get_beta_fit(
    positions: pd.Series, beta_values: pd.Series, beta_err: pd.Series
) -> Tuple[pd.Series, pd.Series]:
    """
    This function returns a fit of the given beta values along with the
    associated error.

    Args:
        positions (pd.Series): Positions of the BPMs to be fitted.
        beta_values (pd.Series): Values of the BPMs to be fitted.
        beta_err (pd.Series): Associated errors to the values.

    Returns:
        Tuple[pd.Series, pd.Series]: The elements returned are the values of
        the fit of the beta values and the associated error.
    """

    def beta_function(x, a, b):
        return a + ((x - b) ** 2) / a

    # Get the rough IP position and beta star for the initial values
    ip_position = (positions[-1] - positions[0]) / 2
    initial_values = (BETA_STAR_ESTIMATION, ip_position)

    # Get the curve fit for the expected 1parabola
    valid = ~(np.isnan(positions) | np.isnan(beta_values))
    fit, fit_cov = curve_fit(
        beta_function,
        positions[valid],
        beta_values[valid],
        p0=initial_values,
        sigma=beta_err[valid],
        maxfev=1000000,
    )

    # Get the error from the covariance matrix
    fit_err = np.sqrt(np.diag(fit_cov))

    # Get the fitted beta and add the errors to get min/max values
    beta_fit = beta_function(positions, fit[0], fit[1])
    beta_max_fit = beta_function(positions, fit[0] + fit_err[0], fit[1] + fit_err[1])
    beta_min_fit = beta_function(positions, fit[0] - fit_err[0], fit[1] - fit_err[1])
    beta_fit_err = (beta_max_fit - beta_min_fit) / 2

    return beta_fit, beta_fit_err


def _get_factors_from_phase(
    beta_phase: pd.Series,
    beta_amp: pd.Series,
    beta_phase_err: pd.Series,
    beta_amp_err: pd.Series,
) -> Tuple[pd.Series, pd.Series]:
    """
    This function computes the calibration factors for the beta method with the
    beta from phase values. The associated error is also calculated.
    The equations being the same for the factors from phase and phase fit, this
    function can be used for both.

    Args:
      beta_phase (pd.Series): Series of the beta from phase values
      beta_amp (pd.Series): Series of the beta from amplitude values
      beta_phase_err (pd.Series): Series of the error associated to the beta from phase values
      beta_amp_err (pd.Series): Series of the error associated to the beta from amplitude values

    Returns:
      Tuple[pd.Series, pd.Series]: The first Series are the calibration
      factors, the second one their error.
    """
    # Compute the calibration factors
    factors = np.sqrt(beta_phase / beta_amp)

    # Now compute the errors
    calibration_error = (beta_phase_err ** 2) / (4 * beta_amp * beta_phase)
    calibration_error += (beta_phase * (beta_amp_err ** 2)) / (4 * (beta_amp ** 3))
    calibration_error = np.sqrt(calibration_error)

    return factors, calibration_error


def get_calibration_factors_from_beta(
    ips: List[int], input_path: Path
) -> Dict[str, pd.DataFrame]:
    """
    This function is the main function to compute the calibration factors for
    the beta method.
    Given a list of IPs and the path containing the corresponding Tfs files,
    this function returns the calibration factors using both the beta from
    phase and its fitted values.

    Args:
      ips (List[int]): IPs to compute the calibration factors for.
      input_path (Path): Path of the directory containing the beta files.

    Returns:
       Dict[str, pd.DataFrame]: The returned DataFrame object contains the
       calibration factors for each BPM along with their error. Both the beta
       from phase and beta from phase fitted values are used, resulting in 6
       colums:
         - NAME: BPM Name
         - S: Position
         - CALIBRATION: Calibration factors computed from beta from phase
         - ERROR_CALIBRATION: Associated error to the above calibration
           factors
         - CALIBRATION_FIT: Calibration factor computed from fitted beta
           from phase
         - ERROR_CALIBRATION_FIT: Associated error to the above calibration
           factors
    """
    LOG.info("Computing the calibration factors via beta")
    # Loop over each plane and compute the calibration factors
    calibration_factors = dict()
    for plane in PLANES:
        LOG.info(f"  Computing the calibration factors for plane {plane}")

        # Load the tfs files for beta from phase and beta from amp
        beta_phase_tfs = tfs.read(
            input_path / f"{BETA_NAME}{plane.lower()}{EXT}", index=TFS_INDEX
        )
        beta_amp_tfs = tfs.read(
            input_path / f"{AMP_BETA_NAME}{plane.lower()}{EXT}", index=TFS_INDEX
        )

        # Get the beam concerned by those tfs files
        beam = int(beta_phase_tfs.iloc[0].name[-1])

        for ip in ips:
            LOG.info(f"    Computing the calibration factors for IP {ip}")
            # Filter our TFS files to only keep the BPMs for the selected IR
            bpms = beta_phase_tfs.reindex(BPMS[ip][beam])
            bpms_amp = beta_amp_tfs.reindex(BPMS[ip][beam])

            # Check for possible missing bpms
            for bpm_set in [bpms, bpms_amp]:
                missing = set(bpm_set.loc[bpm_set.isnull().values].index)
                if missing:
                    LOG.warning("    One or several BPMs are missing in the input"
                                " DataFrame, the calibration factors calculation"
                                f"from fit may not be accurate: {missing}")

            
            # Get the positions and the beta values for those BPMs
            bpms = bpms.index
            positions = beta_phase_tfs.reindex(bpms)[S]
            beta_phase = beta_phase_tfs.reindex(bpms)[f"{BETA}{plane}"]
            beta_phase_err = beta_phase_tfs.reindex(bpms)[f"{ERR}{BETA}{plane}"]
            beta_amp = beta_amp_tfs.reindex(bpms)[f"{BETA}{plane}"]
            beta_amp_err = beta_amp_tfs.reindex(bpms)[f"{ERR}{BETA}{plane}"]

            # Curve fit the beta from phase values
            beta_phase_fit, beta_phase_fit_err = _get_beta_fit(
                positions, beta_phase, beta_phase_err
            )

            # Get the calibration factors for each method: from phase and phase fit
            calibration_phase, calibration_phase_err = _get_factors_from_phase(
                beta_phase, beta_amp, beta_phase_err, beta_amp_err
            )
            calibration_phase_fit, calibration_phase_fit_err = _get_factors_from_phase(
                beta_phase_fit, beta_amp, beta_phase_fit_err, beta_amp_err
            )

            # Assemble the calibration factors in one dataframe
            factors_for_ip = pd.concat(
                [
                    positions,
                    calibration_phase,
                    calibration_phase_err,
                    calibration_phase_fit,
                    calibration_phase_fit_err,
                ],
                axis=1,
            )
            factors_for_ip.columns = LABELS
            factors_for_ip.index.name = TFS_INDEX

            if plane not in calibration_factors.keys():
                calibration_factors[plane] = factors_for_ip
            else:
                calibration_factors[plane] = calibration_factors[plane].append(
                    factors_for_ip
                )

    return calibration_factors
