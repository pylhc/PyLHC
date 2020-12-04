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
    bpms: List[str],
    beta_phase_tfs: pd.DataFrame,
    plane: str
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
    
    positions = beta_phase_tfs.reindex(bpms)[f"{S}"]
    beta_phase = beta_phase_tfs.reindex(bpms)[f"{BETA}{plane}"]
    beta_phase_err = beta_phase_tfs.reindex(bpms)[f"{ERR}{BETA}{plane}"]

    # Get the rough IP position and beta star for the initial values
    ip_position = (positions[-1] - positions[0]) / 2
    initial_values = (BETA_STAR_ESTIMATION, ip_position)

    # Get the curve fit for the expected 1parabola
    valid = ~(np.isnan(positions) | np.isnan(beta_phase))
    popt, pcov = curve_fit(
        beta_function,
        positions[valid],
        beta_phase[valid],
        p0=initial_values,
        sigma=beta_phase_err[valid],
        maxfev=1000000,
        absolute_sigma=True,
    )

    # Get the error from the covariance matrix
    perr = np.sqrt(np.diag(pcov))

    # Get the fitted beta and add the errors to get min/max values
    nstd = 1.
    popt_up = popt + nstd * perr
    popt_dw = popt - nstd * perr

    beta_fit = beta_function(positions[valid], *popt)
    beta_fit_up = beta_function(positions, *popt_up)
    beta_fit_dw = beta_function(positions, *popt_dw)
    beta_fit_err = (beta_fit_up - beta_fit_dw) / 2

    return pd.DataFrame({f"{BETA}{plane}": beta_fit, f"{ERR}{BETA}{plane}": beta_fit_err})


def _get_factors_from_phase(
    beta_phase_tfs: pd.DataFrame,
    beta_amp_tfs: pd.DataFrame,
    plane: str,
) -> Tuple[pd.Series, pd.Series]:
    """
    This function computes the calibration factors for the beta method with the
    beta from phase values. The associated error is also calculated.
    The equations being the same for the factors from phase and phase fit, this
    function can be used for both.

    Args:
      beta_phase (pd.Series): Series of the beta from phase values
      beta_phase_err (pd.Series): Series of the error associated to the beta from phase values
      beta_amp (pd.Series): Series of the beta from amplitude values
      beta_amp_err (pd.Series): Series of the error associated to the beta from amplitude values

    Returns:
      Tuple[pd.Series, pd.Series]: The first Series are the calibration
      factors, the second one their error.
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
    beta_phase_tfs: tfs.TfsDataFrame, 
    beta_amp_tfs: tfs.TfsDataFrame,
    ips: List[int],
    plane: str
) -> Tuple[pd.Series, pd.Series]:
    """
    This function computes the calibration factors for the beta method with the
    beta from phase fit values. The associated error is also calculated.
    This function is used as a wrapper on the `_get_factors_from_fit` function,
    mainly to loop over IPs and filter the BPM list.

    Args:
      beta_phase_tfs (tfs.TfsDataFrame): TfsDataFrame containing all beta from phase values
      beta_amp_tfs (tfs.TfsDataFrame): TfsDataFrame containing all beta from amplitude values
      ips (List[int]): List of IPs to compute the factors for
      beam (int): Beam number
      plane (str): Plane of the BPM measurements

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
                LOG.warning("    One or several BPMs are missing in the input"
                            " DataFrame, the calibration factors calculation"
                            f" from fit may not be accurate: {missing}")

        # Curve fit the beta from phase values
        beta_phase_fit = _get_beta_fit(BPMS[ip][beam], beta_phase_tfs, plane)
    
        # Get the factors and put them all together to have all ips in one
        # Series
        c_fit = _get_factors_from_phase(
            beta_phase_fit,
            beta_amp_tfs.reindex(BPMS[ip][beam]),
            plane
        )
        if calibration_phase_fit is None:
            calibration_phase_fit = c_fit
        else:
            calibration_phase_fit = calibration_phase_fit.append(c_fit)

    # Change the colum names for _fit
    calibration_phase_fit.columns = (LABELS[3], LABELS[4])
    
    return calibration_phase_fit


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
        
        # Get the calibration factors from phase
        calibration_phase = _get_factors_from_phase(beta_phase_tfs, beta_amp_tfs, plane)
        
        # Calibration from phase fit can only be obtained via ballistic optics
        if ips is not None:
            calibration_phase_fit = _get_factors_from_phase_fit(
                beta_phase_tfs, beta_amp_tfs, ips, plane
            )
        else:
            calibration_phase_fit = pd.DataFrame(columns=(LABELS[3], LABELS[4]))

        # Assemble the calibration factors in one dataframe
        factors = pd.concat([beta_phase_tfs[S], calibration_phase, calibration_phase_fit], axis=1)
        factors.columns = LABELS
        factors.index.name = TFS_INDEX
        calibration_factors[plane] = factors

    return calibration_factors
