from pathlib import Path
from scipy.optimize import curve_fit
import numpy as np
import os
import pandas as pd

from generic_parser import EntryPointParameters, entrypoint
from omc3.utils import logging_tools
from omc3.optics_measurements.constants import (
    AMP_BETA_NAME,
    BETA,
    BETA_NAME,
    ERR,
    EXT,
    NORM_DISP_NAME,
    DISPERSION_NAME,
    S,
)
from typing import Tuple, List, Dict

from pylhc.constants.calibration import (
    BPMS,
    BETA_STAR_ESTIMATION,
    CALIBRATION_NAME,
    D_BPMS,
    D,
    IPS,
    LABELS,
    METHODS,
    ND,
    TFS_INDEX,
)
from pylhc.constants.general import BEAMS, PLANES
from tfs.handler import TfsDataFrame
import tfs


LOG = logging_tools.get_logger(__name__)


def _get_params() -> dict:
    """
    Parse Commandline Arguments and return them as options.

    Returns:
        dict
    """

    return EntryPointParameters(
        input_path=dict(
            flags=["--input", "-i"], required=True, type=Path, help="Measurements path."
        ),
        output_path=dict(
            flags=["--outputdir", "-o"],
            type=Path,
            required=True,
            help="Output directory where to write the calibration factors.",
        ),
        ips=dict(
            flags=["--ips"],
            type=int,
            nargs="+",
            choices=IPS,
            required=True,
            help="IPs to compute calibration factors for.",
        ),
        method=dict(
            flags=["--method"],
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


def _get_beta_fit(positions: pd.Series,
                  beta_values: pd.Series, 
                  beta_err: pd.Series) -> Tuple[pd.Series, pd.Series]:
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
    fit, fit_cov = curve_fit(
        beta_function,
        positions,
        beta_values,
        p0=initial_values,
        sigma=beta_err,
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


def _get_factors_from_phase(beta_phase: pd.Series, 
                            beta_amp: pd.Series,
                            beta_phase_err: pd.Series,
                            beta_amp_err: pd.Series) -> Tuple[pd.Series, pd.Series]:
    """
    This function computes the calibration factors for the beta method with the
    beta from phase values. The associated error is also calculated.

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


def _get_factors_from_phase_fit(beta_phase_fit: pd.Series,
                                beta_amp: pd.Series,
                                beta_phase_fit_err: pd.Series,
                                beta_amp_err: pd.Series) -> Tuple[pd.Series, pd.Series]:
    """
    This function computes the calibration factors for the beta method with the
    beta from phase fitted values. The associated error is also calculated.

    Args:
      beta_phase_fit (pd.Series): Series of the beta from phase fitted values
      beta_amp (pd.Series): Series of the beta from amplitude values
      beta_phase_fit_err (pd.Series): Series of the error associated to the beta from phase fitted values
      beta_amp_err (pd.Series): Series of the error associated to the beta from amplitude values

    Returns:
      Tuple[pd.Series, pd.Series]: The first Series are the calibration
      factors, the second one their error.
    """
    # Compute the calibration factors
    factors = np.sqrt(beta_phase_fit / beta_amp)

    # Now compute the errors
    calibration_error = (beta_phase_fit_err ** 2) / (4 * beta_amp * beta_phase_fit)
    calibration_error += (beta_phase_fit * (beta_amp_err ** 2)) / (4 * (beta_amp ** 3))
    calibration_error = np.sqrt(calibration_error)

    return factors, calibration_error


def _get_calibration_factors_from_beta(ips: List[int],
                                       input_path: Path) -> pd.DataFrame:
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
       pd.DataFrame: The returned DataFrame object contains the calibration
       factors for each BPM along with their error. Both the beta from phase
       and beta from phase fitted values are used, resulting in 6 colums: 
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
        # Load the tfs files for beta from phase and beta from amp
        beta_phase_tfs = tfs.read(input_path / f'{BETA_NAME}{plane.lower()}{EXT}', index=TFS_INDEX)
        beta_amp_tfs = tfs.read(input_path / f'{AMP_BETA_NAME}{plane.lower()}{EXT}', index=TFS_INDEX)

        # Get the beam concerned by those tfs files
        beam = int(beta_phase_tfs.iloc[0].name[-1])

        for ip in ips:
            LOG.debug("Computing the calibration factors for IP {ip}")
            # Filter our TFS files to only keep the BPMs for the selected IR
            bpms = beta_phase_tfs.reindex(BPMS[ip][beam]).dropna().index

            # Get the positions and the beta values for those BPMs
            positions = beta_phase_tfs.loc[bpms, S].dropna()
            beta_phase = beta_phase_tfs.loc[bpms, f"{BETA}{plane}"].dropna()
            beta_phase_err = beta_phase_tfs.loc[bpms, f"{ERR}{BETA}{plane}"].dropna()
            beta_amp = beta_amp_tfs.loc[bpms, f"{BETA}{plane}"].dropna()
            beta_amp_err = beta_amp_tfs.loc[bpms, f"{ERR}{BETA}{plane}"].dropna()

            # Curve fit the beta from phase values
            beta_phase_fit, beta_phase_fit_err = _get_beta_fit(
                positions, beta_phase, beta_phase_err
            )

            # Get the calibration factors for each method: from phase and phase fit
            calibration_phase, calibration_phase_err = _get_factors_from_phase(
                beta_phase, beta_amp, beta_phase_err, beta_amp_err
            )
            calibration_phase_fit, calibration_phase_fit_err = _get_factors_from_phase_fit(
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
                calibration_factors[plane] = calibration_factors[plane].append(factors_for_ip)
    
    LOG.debug(f"Calibration factors: {calibration_factors}")

    return calibration_factors


def _get_dispersion_from_phase(normalised_dispersion: Dict[str, pd.Series],
                              beta: Dict[str, pd.Series]) -> Tuple[pd.Series, pd.Series]:
    """
    This function computes the dispersion from phase given the normalised
    dispersion from amplitude, the beta from phase and their associated errors.

    Args:
        normalised_dispersion (Dict[str, pd.Series]): Dictionnary containg the
        keys "amp" and and "amp_err" with a pd.Series item as value for each.
        beta (Dict[str, pd.Series]): Dictionnary containg the keys "phase" and
        and "phase_err" with a pd.Series item as value for each.

    Returns
        Tuple[pd.Series, pd.Series]: The dispersion from phase and its associated error in each Series
    """
    # Compute the dispersion from phase
    d_phase = normalised_dispersion["amp"] * np.sqrt(beta["phase"])

    # And the the associated error
    d_phase_err = (normalised_dispersion["amp_err"] * np.sqrt(beta["phase_err"])) ** 2
    d_phase_err += (
        1 / 2
        * normalised_dispersion["amp"]
        / np.sqrt(beta["phase"])
        * beta["phase_err"]
    ) ** 2
    d_phase_err = np.sqrt(d_phase_err)

    return d_phase, d_phase_err


def _get_dispersion_fit(positions: pd.Series,
                        dispersion_values: pd.Series,
                        dispersion_err: pd.Series) -> Tuple[pd.Series, pd.Series]:
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
    fit, fit_cov = curve_fit(dispersion_function, positions, values, sigma=err)

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


def _get_factors_from_dispersion(dispersion: Dict[str, pd.Series]) -> Tuple[pd.Series, pd.Series]:
    """
    This function computes the calibration factors for the dispersion method
    with the non fitted dispersion values. The associated error is also
    calculated.

    Args:
      dispersion (Dict[str, pd.Series]): Dictionnary containing 4 keys: phase, phase_err, amp and amp_err.
      Each key is related to the method used to obtain the dispersion and its
      error.

    Returns:
      Tuple[pd.Series, pd.Series]: The first Series are the calibration factors, the second one their
      error.
    """
    # Get the ratios, those are our calibration factors
    factors = dispersion["phase"] / dispersion["amp"]

    # Code in BBs
    calibration_error = (dispersion["phase_err"] / dispersion["amp"]) ** 2
    calibration_error += (
        dispersion["amp_err"] * (dispersion["phase"] / (dispersion["amp"] ** 2))
    ) ** 2
    calibration_error = np.sqrt(calibration_error)

    return factors, calibration_error


def _get_factors_from_dispersion_fit(dispersion: Dict[str, pd.Series]) -> Tuple[pd.Series, pd.Series]:
    """
    This function computes the calibration factors for the dispersion method
    with the _fitted_ dispersion values. The associated error is also
    calculated.

    Args:
      dispersion (dict): Dictionnary containing 4 keys: phase_fit,
      phase_fit_err, amp and amp_err.  Each key is related to the method used
      to obtain the dispersion and its error.

    Returns:
      Tuple[pd.Series, pd.Series]: The first Series are the calibration
      factors, the second one their error.
    """
    # Get the ratios, those are our calibration factors
    factors = dispersion["phase_fit"] / dispersion["amp"]

    calibration_error = (dispersion["phase_fit_err"] / dispersion["amp"]) ** 2
    calibration_error += (
        dispersion["amp_err"] * dispersion["phase_fit"] / (dispersion["amp"] ** 2)
    ) ** 2
    calibration_error = np.sqrt(calibration_error)

    return factors, calibration_error


def _get_calibration_factors_from_dispersion(ips: List[int],
                                             input_path: Path) -> pd.DataFrame:
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
      pd.DataFrame: The returned DataFrame object contains the calibration
      factors for each BPM along with their error. Both the dispersion and
      dispersion from fit values are used, resulting in 6 colums: 
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
    norm_dispersion_tfs = tfs.read(input_path / f'{NORM_DISP_NAME}x{EXT}', index=TFS_INDEX)
    dispersion_tfs = tfs.read(input_path / f'{DISPERSION_NAME}x{EXT}', index=TFS_INDEX)
    
    # Get the beam concerned by those tfs files
    beam = int(dispersion_tfs.iloc[0].name[-1])

    # Loop over the IPs and compute the calibration factors
    calibration_factors = dict()
    for ip in ips:
        LOG.debug("Computing the calibration factors for IP {ip}")
        # Filter our TFS files to only keep the BPMs for the selected IR
        bpms = dispersion_tfs.reindex(BPMS[ip][beam]).dropna().index
        d_bpms = dispersion_tfs.reindex(D_BPMS[ip][beam]).dropna().index

        # Get the positions of the BPMs and the subset used for the fit
        positions = dispersion_tfs.loc[bpms, S].dropna()
        positions_fit = dispersion_tfs.loc[d_bpms, S].dropna()

        # Get the dispersion and normalised dispersion from the tfs files
        dispersion = dict()
        normalised_dispersion = dict()

        dispersion["amp"] = dispersion_tfs.loc[bpms, f"DX"].dropna()
        dispersion["amp_err"] = dispersion_tfs.loc[bpms, f"{ERR}{D}X"].dropna()

        dispersion["phase"] = norm_dispersion_tfs.loc[bpms, f"DX"].dropna()
        dispersion["phase_err"] = norm_dispersion_tfs.loc[bpms, f"{ERR}{D}X"].dropna()

        normalised_dispersion["amp"] = norm_dispersion_tfs.loc[bpms, f"{ND}X"].dropna()
        normalised_dispersion["amp_err"] = norm_dispersion_tfs.loc[
            bpms, f"{ERR}{ND}X"
        ].dropna()

        # Compute the calibration factors using the dispersion from phase and amp
        calibration, calibration_err = _get_factors_from_dispersion(dispersion)

        # Fit the dispersion from phase
        dispersion["phase_fit"], dispersion["phase_fit_err"] = _get_dispersion_fit(
            positions_fit, dispersion["phase"], dispersion["phase_err"]
        )

        # Compute the calibration factors using the fitted dispersion from amp / phase
        calibration_fit, calibration_fit_err = _get_factors_from_dispersion_fit(dispersion)

        # Assemble the calibration factors in one dataframe
        factors_for_ip = pd.concat(
            [positions, calibration, calibration_err, calibration_fit, calibration_fit_err],
            axis=1,
        )
        factors_for_ip.columns = LABELS
        factors_for_ip.index.name = TFS_INDEX

        if 'X' not in calibration_factors.keys():
            calibration_factors = {'X': factors_for_ip}
        else:
            calibration_factors['X'] = calibration_factors['X'].append(factors_for_ip)

    LOG.debug(f"Calibration factors: {calibration_factors}")
    
    return calibration_factors


def _write_calibration_tfs(calibration_factors: pd.DataFrame,
                          plane: str,
                          method: str, 
                          output_path: Path) -> None:
    """
    This function saves to a file the calibration factors given as input.
    The file name contains the method and plane related to those factors.
    e.g: "calibration_beta_x.tfs"

    Args:
      calibration_factors (pd.DataFrame): The DataFrame containing the calibration factors to be written to disk.
      plane (str): The plane associated to the current calibration factors.
      method (str): The method used to compute those calibration factors.
      output_path (Path): The directory where to save the file.

    Returns:
      None
    """
    # Create the output directory
    os.makedirs(output_path.absolute(), exist_ok=True)

    # Reset the index of the dataframe, it was handy to handle the data but not
    # to store it
    calibration_factors = calibration_factors.reset_index()

    # Write the TFS files for this plane
    # The method chosen will change the tfs name
    tfs_name = f"{CALIBRATION_NAME[method]}{plane.lower()}{EXT}"
    LOG.info(f"Writing {tfs_name}")
    file_path = output_path / tfs_name
    tfs.write_tfs(file_path, calibration_factors, save_index=False)


@entrypoint(_get_params(), strict=True)
def main(opt):
    # Compute the calibration factors and their errors according to the method
    if opt.method == "beta":
        factors = _get_calibration_factors_from_beta(opt.ips, opt.input_path)
    elif opt.method == "dispersion":
        factors = _get_calibration_factors_from_dispersion(opt.ips, opt.input_path)

    # Write the TFS file to the desired output directory
    for plane in factors.keys():
        _write_calibration_tfs(factors[plane], plane, opt.method, opt.output_path)

    return factors


if __name__ == "__main__":
    main()
