"""
Beta
-----

The functions in this script compute the calibration factors for the LHC BPMs
using the beta method. The `get_calibration_factors_from_beta` is intended
to be used with the script `bpm_calibration.py`.

"""
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd
import tfs
from omc3.optics_measurements.constants import AMP_BETA_NAME, BETA, BETA_NAME, ERR, EXT, S
from omc3.utils import logging_tools
from scipy.optimize import curve_fit

from pylhc.constants.calibration import BETA_STAR_ESTIMATION, BPMS, IPS, LABELS, TFS_INDEX
from pylhc.constants.general import PLANES

LOG = logging_tools.get_logger(__name__)


def _get_beta_fit(
    bpms: Sequence[str],
    beta_phase_tfs: pd.DataFrame,
    plane: str,
) -> pd.DataFrame:
    """
    This function returns a fit of the given beta values along with the
    associated error.

    Args:
        bpms (Sequence[str]): Names of the BPMs to use for the fitting.
        beta_phase_tfs (pd.DataFrame): A ``DataFrame`` with beta from phase values to be fitted at the
            given BPMs.
        plane (str): plane to perform the fit on.

    Returns:
        A ``pandas.DataFrame`` with the resulting values from the fit of the input beta values and the
        associated errors, as columns.
    """

    def beta_function(x, a, b):
        return a + ((x - b) ** 2) / a

    def err_function(x, popt, pcov):
        sa, sb, sab = pcov[0, 0], pcov[1, 1], pcov[0, 1]
        a, b = popt[0], popt[1]

        beta_err = ((a ** 2 - (x - b) ** 2) / a ** 2) ** 2 * sa
        beta_err += 4 * ((x - b) / a) ** 2 * sb
        beta_err -= 4 * (x - b) * (a ** 2 - (x - b) ** 2) / a ** 3 * sab
        return beta_err

    positions = beta_phase_tfs.reindex(bpms)[f"{S}"]
    beta_phase = beta_phase_tfs.reindex(bpms)[f"{BETA}{plane}"]
    beta_phase_err = beta_phase_tfs.reindex(bpms)[f"{ERR}{BETA}{plane}"]

    # Get the rough IP position and beta star for the initial values
    ip_position = (positions[-1] - positions[0]) / 2
    initial_values = (BETA_STAR_ESTIMATION, ip_position)

    # Get the curve fit for the expected parabola
    valid = ~(np.isnan(positions) | np.isnan(beta_phase))

    additional_args = {}
    if sum(beta_phase_err[valid]) != 0:
        additional_args = {"sigma": beta_phase_err[valid]}

    popt, pcov = curve_fit(
        beta_function,
        positions[valid],
        beta_phase[valid],
        p0=initial_values,
        maxfev=1000000,
        absolute_sigma=True,
        **additional_args,
    )

    # Get the error from the covariance matrix
    perr = np.sqrt(np.diag(pcov))

    # Get the fitted beta and add the errors to get min/max values
    beta_fit = beta_function(positions[valid], *popt)
    beta_fit_err = err_function(positions[valid], popt, pcov)

    return pd.DataFrame({f"{BETA}{plane}": beta_fit, f"{ERR}{BETA}{plane}": beta_fit_err})


def _get_factors_from_phase(
    beta_phase_tfs: pd.DataFrame,
    beta_amp_tfs: pd.DataFrame,
    plane: str,
) -> pd.DataFrame:
    """
    This function computes the calibration factors for the beta method with the
    beta from phase values. The associated error is also calculated.
    The equations being the same for the factors from phase and phase fit, this
    function can be used for both.

    Args:
        beta_phase_tfs (pd.DataFrame): A ``DataFrame`` with beta from phase values.
        beta_amp_tfs (pd.DataFrame): A ``DataFrame`` with beta from amplitude values.
        plane (str): Plane of the BPM measurements.

    Returns:
        A ``pandas.DataFrame`` with the computed calibration factors and their associated errors as columns.
    """
    beta_phase = beta_phase_tfs[f"{BETA}{plane}"]
    beta_phase_err = beta_phase_tfs[f"{ERR}{BETA}{plane}"]
    beta_amp = beta_amp_tfs[f"{BETA}{plane}"]
    beta_amp_err = beta_amp_tfs[f"{ERR}{BETA}{plane}"]

    # Compute the calibration factors
    factors = np.sqrt(beta_phase / beta_amp)

    # Now compute the errors
    calibration_error = (beta_phase_err ** 2) / (4 * beta_amp * beta_phase)
    calibration_error += (beta_phase * (beta_amp_err ** 2)) / (4 * (beta_amp ** 3))
    calibration_error = np.sqrt(calibration_error)

    return pd.DataFrame({LABELS[1]: factors, LABELS[2]: calibration_error})


def _get_factors_from_phase_fit(
    beta_phase_tfs: pd.DataFrame,
    beta_amp_tfs: pd.DataFrame,
    ips: Sequence[int],
    plane: str,
) -> Tuple[pd.Series, pd.Series]:
    """
    This function computes the calibration factors for the beta method with the
    beta from phase fit values. The associated error is also calculated.
    This function is used as a wrapper on the `_get_factors_from_fit` function,
    mainly to loop over IPs and filter the BPM list.

    Args:
        beta_phase_tfs (pd.DataFrame): A ``DataFrame`` with beta from phase values.
        beta_amp_tfs (pd.DataFrame): A ``DataFrame`` with beta from amplitude values.
        ips (Sequence[int]): List of IPs to compute the factors for.
        plane (str): Plane of the BPM measurements.

    Returns:
      Tuple[pd.Series, pd.Series]: The first Series are the calibration
      factors, the second one their error.
    """
    # Get the beam concerned by those tfs files
    beam = int(beta_phase_tfs.iloc[0].name[-1])

    calibration_phase_fit, calibration_phase_fit_err = None, None
    for ip in ips:
        LOG.info(f"    Computing the calibration factors from phase fit for IP {ip}")

        # Check for possible missing bpms
        bpms = beta_phase_tfs.reindex(BPMS[ip][beam])
        bpms_amp = beta_amp_tfs.reindex(BPMS[ip][beam])
        for bpm_set in [bpms, bpms_amp]:
            missing = set(bpm_set.loc[bpm_set.isnull().values].index)
            if missing:
                LOG.warning(
                    "    One or several BPMs are missing in the input"
                    " DataFrame, the calibration factors calculation"
                    f" from fit may not be accurate: {missing}"
                )

        # Curve fit the beta from phase values
        beta_phase_fit = _get_beta_fit(BPMS[ip][beam], beta_phase_tfs, plane)

        # Get the factors and put them all together to have all ips in one
        # Series
        c_fit = _get_factors_from_phase(beta_phase_fit, beta_amp_tfs.reindex(BPMS[ip][beam]), plane)
        if calibration_phase_fit is None:
            calibration_phase_fit = c_fit
        else:
            calibration_phase_fit = calibration_phase_fit.append(c_fit)

    # Change the colum names for _fit
    calibration_phase_fit.columns = (LABELS[3], LABELS[4])

    return calibration_phase_fit


def get_calibration_factors_from_beta(ips: Sequence[int], input_path: Path) -> Dict[str, pd.DataFrame]:
    """
    This function is the main function to compute the calibration factors for
    the beta method.
    Given a list of IPs and the path containing the corresponding Tfs files,
    this function returns the calibration factors using both the beta from
    phase and its fitted values.

    Args:
      ips (Sequence[int]): IPs to compute the calibration factors for.
      input_path (Path): Path of the directory containing the beta files.

    Returns:
       Dict[str, pd.DataFrame]: The returned DataFrame object contains the
       calibration factors for each BPM along with their error. Both the beta
       from phase and beta from phase fitted values are used, resulting in 6
       columns:

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
        beta_phase_tfs = tfs.read(input_path / f"{BETA_NAME}{plane.lower()}{EXT}", index=TFS_INDEX)
        beta_amp_tfs = tfs.read(input_path / f"{AMP_BETA_NAME}{plane.lower()}{EXT}", index=TFS_INDEX)

        # Get the calibration factors from phase
        calibration_phase = _get_factors_from_phase(beta_phase_tfs, beta_amp_tfs, plane)

        # Calibration from phase fit can only be obtained via ballistic optics
        if ips is not None:
            calibration_phase_fit = _get_factors_from_phase_fit(beta_phase_tfs, beta_amp_tfs, ips, plane)
        else:
            calibration_phase_fit = pd.DataFrame(columns=(LABELS[3], LABELS[4]))

        # Assemble the calibration factors in one dataframe
        factors = pd.concat([beta_phase_tfs[S], calibration_phase, calibration_phase_fit], axis=1)
        factors.columns = LABELS
        factors.index.name = TFS_INDEX
        calibration_factors[plane] = factors

    return calibration_factors
