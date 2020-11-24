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
    S,
)
from pylhc.constants.calibration import (
    BPMS,
    BETA_STAR_ESTIMATION,
    CALIBRATION_NAME,
    D_BPMS,
    D,
    IPS,
    LABELS,
    METHODS,
    MODEL_TFS,
    ND,
    TFS_INDEX,
)
from pylhc.constants.general import BEAMS, PLANES
from tfs.handler import TfsDataFrame
import tfs



LOG = logging_tools.get_logger(__name__)


def _get_params() -> dict:
    """_get_params.

    Parse Commandline Arguments and return them as options.

    Returns
    -------
    dict

    """

    return EntryPointParameters(
        input_path=dict(
            flags=["--input", "-i"], required=True, type=Path, help="Measurements path."
        ),
        model_path=dict(
            flags=["--model", "-m"],
            type=Path,
            required=True,
            help="Model path associated to the measurements.",
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


def _get_tfs_files(prefix: str, planes: list, input_path: Path) -> dict:
    """_get_tfs_files.

    This function loops over the planes given as input and will look for the
    files matching the filename "prefix + plane.tfs".
    Each plane is then associated to its tfs object in a dict.

    Parameters
    ----------
    prefix : str
        Prefix of the file to be opened. e.g. 'BET'
    planes : list
        List of planes to consider when opening files
    input_path : Path
        Directory where the file is located

    Returns
    -------
    dict
        A dictionnary associating one plane to one tfs object
    """
    tfs_files = dict()
    for plane in planes:
        file_name = f"{prefix}{plane.lower()}{EXT}"
        expected_path = input_path / file_name

        if not expected_path.is_file():
            msg = f"File {file_name} couldn't be found in directory {input_path}"
            LOG.error(msg)
            raise FileNotFoundError(msg)

        tfs_files[plane] = tfs.read(expected_path, index=TFS_INDEX)
    return tfs_files


def _get_beta_from_phase_tfs(input_path: Path) -> dict:
    """_get_beta_from_phase_tfs.
    
    This function returns the beta from phase file associated to the
    measurement's directory given as input. Both planes X and Y are used.

    Parameters
    ----------
    input_path : Path
        Directory where the beta from phase files are located.

    Returns
    -------
    dict
        A dictionnary with two keys, X and Y and the associated tfs objects for
        each plane.

    """
    return _get_tfs_files(BETA_NAME, PLANES, input_path)


def _get_beta_from_amp_tfs(input_path: Path) -> dict:
    """_get_beta_from_amp_tfs.
    
    This function returns the beta from amplitude file associated to the
    measurement's directory given as input. Both planes X and Y are used.

    Parameters
    ----------
    input_path : Path
        Directory where the beta from amplitude files are located.

    Returns
    -------
    dict
        A dictionnary with two keys, X and Y and the associated tfs objects for
        each plane.

    """
    return _get_tfs_files(AMP_BETA_NAME, PLANES, input_path)


def _get_dispersion_tfs(input_path: Path) -> dict:
    """_get_dispersion_tfs.

    This function returns the tfs dispersion file associated to the
    measurement's directory given as input. Only the X plane is used.

    Parameters
    ----------
    input_path : Path
        Directory where the dispersion file for the X plane is located.

    Returns
    -------
    dict
        A dictionnary with one key, X, and its associated tfs object.
    """
    return _get_tfs_files(NORM_DISP_NAME, "X", input_path)


def _get_beam_from_model(model_tfs: TfsDataFrame) -> int:
    """_get_beam_from_model.

    Given a tfs object of the model, this function returns which beam it is
    associated to.

    Parameters
    ----------
    model_tfs : TfsDataFrame
        tfs object of the model currently used.

    Returns
    -------
    int
        Beam number.

    """
    beam = int(model_tfs.SEQUENCE[-1])
    if beam not in BEAMS:
        msg = f"Could not find a correct value for beam in model: {beam}"
        LOG.error(msg)
        raise ValueError(msg)
    return beam


def _get_beta_fit(positions: pd.Series, 
                  beta_values: pd.Series, 
                  beta_err: pd.Series) -> (pd.Series, pd.Series):
    """_get_beta_fit.

    This function returns a fit of the given beta values along with the
    associated error.

    Parameters
    ----------
    positions : pd.Series
        Positions of the BPMs to be fitted.
    beta_values : pd.Series
        Values of the BPMs to be fitted.
    beta_err : pd.Series
        Associated errors to the values.

    Returns
    -------
    (pd.Series, pd.Series)
        The elements returned are the values of the fit of the beta values
        and the associated error.

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
                            beta_amp_err: pd.Series) -> (pd.Series, pd.Series):
    """_get_factors_from_phase.

    This function computes the calibration factors for the beta method with the
    beta from phase values. The associated error is also calculated.

    Parameters
    ----------
    beta_phase : pd.Series
        Series of the beta from phase values
    beta_amp : pd.Series
        Series of the beta from amplitude values
    beta_phase_err : pd.Series
        Series of the error associated to the beta from phase values
    beta_amp_err : pd.Series
        Series of the error associated to the beta from amplitude values

    Returns
    -------
    (pd.Series, pd.Series)
        The first Series are the calibration factors, the second one their
        error.

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
                                beta_amp_err: pd.Series) -> (pd.Series, pd.Series):
    """_get_factors_from_phase_fit.

    This function computes the calibration factors for the beta method with the
    beta from phase fitted values. The associated error is also calculated.

    Parameters
    ----------
    beta_phase_fit : pd.Series
        Series of the beta from phase fitted values
    beta_amp : pd.Series
        Series of the beta from amplitude values
    beta_phase_fit_err : pd.Series
        Series of the error associated to the beta from phase fitted values
    beta_amp_err : pd.Series
        Series of the error associated to the beta from amplitude values

    Returns
    -------
    (pd.Series, pd.Series)
        The first Series are the calibration factors, the second one their
        error.

    """
    # Compute the calibration factors
    factors = np.sqrt(beta_phase_fit / beta_amp)

    # Now compute the errors
    calibration_error = (beta_phase_fit_err ** 2) / (4 * beta_amp * beta_phase_fit)
    calibration_error += (beta_phase_fit * (beta_amp_err ** 2)) / (4 * (beta_amp ** 3))
    calibration_error = np.sqrt(calibration_error)

    return factors, calibration_error


def _get_calibration_factors_beta(ip: int,
                                  plane: str,
                                  beam: int,
                                  beta_phase_tfs: TfsDataFrame,
                                  beta_amp_tfs: TfsDataFrame) -> pd.DataFrame:
    """_get_calibration_factors_beta.

    This function is the main function to compute the calibration factors for
    the beta method.
    Given an IP, a plane and the corresponding Tfs files, this function
    returns the calibration factors using both the beta from phase and its
    fitted values.


    Parameters
    ----------
    ip: int
        IP to compute the calibration factors for.
    plane : str
        Plane to compute the calibration factors for.
    beam: int
        Beam to compute the calibration factors for.
    beta_phase_tfs: TfsDataFrame
        Tfs object for the beta from phase values.
    beta_amp_tfs: TfsDataFrame
        Tfs object for the beta from amplitude values.

    Returns
    -------
    pd.DataFrame
        The returned DataFrame object contains the calibration factors for each
        BPM along with their error.
        Both the beta from phase and beta from phase fitted values are used,
        resulting in 6 colums: 
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
    calibration_factors = pd.concat(
        [
            positions,
            calibration_phase,
            calibration_phase_err,
            calibration_phase_fit,
            calibration_phase_fit_err,
        ],
        axis=1,
    )
    calibration_factors.columns = LABELS
    calibration_factors.index.name = TFS_INDEX

    return calibration_factors


def _get_dispersion_from_phase(normalised_dispersion: dict,
                              beta: dict) -> (pd.Series, pd.Series):
    """_get_dispersion_from_phase.

    This function computes the dispersion from phase given the normalised
    dispersion from amplitude, the beta from phase and their associated errors.

    Parameters
    ----------
    normalised_dispersion : dict
        Dictionnary containg the keys "amp" and and "amp_err" with a pd.Series
        item as value for each.
        
    beta : dict
        Dictionnary containg the keys "phase" and and "phase_err" with a
        pd.Series item as value for each.

    Returns
    -------
    (pd.Series, pd.Series)
        The dispersion from phase and its associated error in each Series

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
                        dispersion_err: pd.Series) -> (pd.Series, pd.Series):
    """_get_dispersion_fit.

    This function returns a fit of the given dispersion values along with the
    associated error.

    Parameters
    ----------
    positions : pd.Series
        Positions of the BPMs to be fitted.
    dispersion_values : pd.Series
        Values of the BPMs to be fitted.
    dispersion_err : pd.Series
        Associated errors to the values.

    Returns
    -------
    (pd.Series, pd.Series)
        The elements returned are the values of the fit of the dispersion
        values and the associated error.

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


def _get_factors_from_dispersion(dispersion: dict) -> (pd.Series, pd.Series):
    """_get_factors_from_dispersion.

    This function computes the calibration factors for the dispersion method
    with the non fitted dispersion values. The associated error is also
    calculated.

    Parameters
    ----------
    dispersion: dict
        Dictionnary containing 4 keys: phase, phase_err, amp and amp_err.
        Each key is related to the method used to obtain the dispersion and its
        error.

    Returns
    -------
    (pd.Series, pd.Series)
        The first Series are the calibration factors, the second one their
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


def _get_factors_from_dispersion_fit(dispersion: dict) -> (pd.Series, pd.Series):
    """_get_factors_from_dispersion_fit.

    This function computes the calibration factors for the dispersion method
    with the _fitted_ dispersion values. The associated error is also
    calculated.

    Parameters
    ----------
    dispersion: dict
        Dictionnary containing 4 keys: phase_fit, phase_fit_err, amp and 
        amp_err.
        Each key is related to the method used to obtain the dispersion and its
        error.

    Returns
    -------
    (pd.Series, pd.Series)
        The first Series are the calibration factors, the second one their
        error.
    """
    # Get the ratios, those are our calibration factors
    factors = dispersion["phase_fit"] / dispersion["amp"]

    calibration_error = (dispersion["phase_fit_err"] / dispersion["amp"]) ** 2
    calibration_error += (
        dispersion["amp_err"] * dispersion["phase_fit"] / (dispersion["amp"] ** 2)
    ) ** 2
    calibration_error = np.sqrt(calibration_error)

    return factors, calibration_error


def _get_calibration_factors_dispersion(ip: int, 
                                        plane: str,
                                        beam: int,
                                        beta_phase_tfs: TfsDataFrame,
                                        dispersion_tfs: TfsDataFrame) -> pd.DataFrame:
    """_get_calibration_factors_beta.

    This function is the main function to compute the calibration factors for
    the dispersion method.
    Given an IP, a plane and the corresponding Tfs files, this function
    returns the calibration factors using both the dispersion and its fitted
    values.


    Parameters
    ----------
    ip: int
        IP to compute the calibration factors for.
    plane : str
        Plane to compute the calibration factors for.
    beam: int
        Beam to compute the calibration factors for.
    beta_phase_tfs: TfsDataFrame
        Tfs object for the beta from phase values.
    dispersion_tfs: TfsDataFrame
        Tfs object for the dispersion values.

    Returns
    -------
    pd.DataFrame
        The returned DataFrame object contains the calibration factors for each
        BPM along with their error.
        Both the dispersion and dispersion from fit values are used,
        resulting in 6 colums: 
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
    # Filter our TFS files to only keep the BPMs for the selected IR
    bpms = beta_phase_tfs.reindex(BPMS[ip][beam]).dropna().index
    d_bpms = beta_phase_tfs.reindex(D_BPMS[ip][beam]).dropna().index

    # Get the positions of the BPMs and the subset used for the fit
    positions = beta_phase_tfs.loc[bpms, S].dropna()
    positions_fit = beta_phase_tfs.loc[d_bpms, S].dropna()

    # Get the beta values for all BPMs
    beta = dict()
    beta["phase"] = beta_phase_tfs.loc[bpms, f"{BETA}{plane}"].dropna()
    beta["phase_err"] = beta_phase_tfs.loc[bpms, f"{ERR}{BETA}{plane}"].dropna()

    # Get the dispersion and normalised dispersion from the tfs files
    dispersion = dict()
    normalised_dispersion = dict()
    dispersion["amp"] = dispersion_tfs.loc[bpms, f"D{plane}"].dropna()
    dispersion["amp_err"] = dispersion_tfs.loc[bpms, f"{ERR}{D}{plane}"].dropna()
    normalised_dispersion["amp"] = dispersion_tfs.loc[bpms, f"{ND}{plane}"].dropna()
    normalised_dispersion["amp_err"] = dispersion_tfs.loc[
        bpms, f"{ERR}{ND}{plane}"
    ].dropna()

    # Get the dispersion from phase
    dispersion["phase"], dispersion["phase_err"] = _get_dispersion_from_phase(
        normalised_dispersion, beta
    )

    # Compute the calibration factors using the dispersion from phase and amp
    calibration, calibration_err = _get_factors_from_dispersion(dispersion)

    # Fit the dispersion from phase
    dispersion["phase_fit"], dispersion["phase_fit_err"] = _get_dispersion_fit(
        positions_fit, dispersion["phase"], dispersion["phase_err"]
    )

    # Compute the calibration factors using the fitted dispersion from amp / phase
    calibration_fit, calibration_fit_err = _get_factors_from_dispersion_fit(dispersion)

    # Assemble the calibration factors in one dataframe
    calibration_factors = pd.concat(
        [positions, calibration, calibration_err, calibration_fit, calibration_fit_err],
        axis=1,
    )

    calibration_factors.columns = LABELS
    calibration_factors.index.name = TFS_INDEX
    
    return calibration_factors


def _write_calibration_tfs(calibration_factors: pd.DataFrame,
                          plane: str,
                          method: str, 
                          output_path: Path) -> None:
    """_write_calibration_tfs.

    This function saves to a file the calibration factors given as input.
    The file name contains the method and plane related to those factors.
    e.g: "calibration_beta_x.tfs"

    Parameters
    ----------
    calibration_factors : pd.DataFrame
        The DataFrame containing the calibration factors to be written to disk.
    plane : str
        The plane associated to the current calibration factors.
    method : str
        The method used to compute those calibration factors.
    output_path : Path
        The directory where to save the file.

    Returns
    -------
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
    file_path = output_path / tfs_name
    tfs.write_tfs(file_path, calibration_factors, save_index=False)


@entrypoint(_get_params(), strict=True)
def main(opt):
    # Load the tfs for beta from phase and beta from amp
    beta_phase_tfs = _get_beta_from_phase_tfs(opt.input_path)
    beta_amp_tfs = _get_beta_from_amp_tfs(opt.input_path)

    # Get the beam number from the model
    model_tfs = tfs.read(opt.model_path / MODEL_TFS, index=TFS_INDEX)
    beam = _get_beam_from_model(model_tfs)

    # also load the dispersion file if the method requires it
    if opt.method == "dispersion":
        dispersion_tfs = _get_dispersion_tfs(opt.input_path)["X"]

    # Compute the calibration factors and their errors for each plane
    c_factors = dict()
    for plane in PLANES:
        c_factors[plane] = tfs.TfsDataFrame()
        for ip in opt.ips:
            if opt.method == "beta":
                factors = _get_calibration_factors_beta(
                    ip, plane, beam, beta_phase_tfs[plane], beta_amp_tfs[plane]
                )
            elif opt.method == "dispersion" and plane == "X":
                factors = _get_calibration_factors_dispersion(
                    ip, plane, beam, beta_phase_tfs[plane], dispersion_tfs
                )

            c_factors[plane] = c_factors[plane].append(factors)

        # Write the TFS file to the desired output directory
        # There's no calibration factors on the Y plane for dispersion
        if opt.method == "dispersion" and plane != "X":
            continue
        _write_calibration_tfs(c_factors[plane], plane, opt.method, opt.output_path)

    return c_factors


if __name__ == "__main__":
    main()
