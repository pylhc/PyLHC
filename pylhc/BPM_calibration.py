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
import tfs


LOG = logging_tools.get_logger(__name__)


def _get_params() -> dict:
    """ Parse Commandline Arguments and return them as options. """

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


def _get_tfs_files(prefix, planes, input_path):
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


def _get_beta_from_phase_tfs(input_path):
    return _get_tfs_files(BETA_NAME, PLANES, input_path)


def _get_beta_from_amp_tfs(input_path):
    return _get_tfs_files(AMP_BETA_NAME, PLANES, input_path)


def _get_dispersion_tfs(input_path):
    return _get_tfs_files(NORM_DISP_NAME, "X", input_path)


def _get_beam_from_model(model_tfs):
    beam = int(model_tfs.SEQUENCE[-1])
    if beam not in BEAMS:
        msg = f"Could not find a correct value for beam in model: {beam}"
        LOG.error(msg)
        raise ValueError(msg)
    return beam


def _get_beta_fit(positions, beta_values, beta_err):
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
    beta_star_fit_err = (beta_max_fit - beta_min_fit) / 2

    return beta_fit, beta_star_fit_err


def _get_factors_from_phase(beta_phase, beta_amp, beta_phase_err, beta_amp_err):
    # Compute the calibration factors
    factors = np.sqrt(beta_phase / beta_amp)

    # Now compute the errors
    calibration_error = (beta_phase_err ** 2) / (4 * beta_amp * beta_phase)
    calibration_error += (beta_phase * (beta_amp_err ** 2)) / (4 * (beta_amp ** 3))
    calibration_error = np.sqrt(calibration_error)

    return factors, calibration_error


def _get_factors_from_phase_fit(
    beta_phase_fit, beta_amp, beta_phase_fit_err, beta_amp_err
):
    # Compute the calibration factors
    factors = np.sqrt(beta_phase_fit / beta_amp)

    # Now compute the errors
    calibration_error = (beta_phase_fit_err ** 2) / (4 * beta_amp * beta_phase_fit)
    calibration_error += (beta_phase_fit * (beta_amp_err ** 2)) / (4 * (beta_amp ** 3))
    calibration_error = np.sqrt(calibration_error)

    return factors, calibration_error


def _get_calibration_factors_beta(ip, plane, beta_phase_tfs, beta_amp_tfs, model_tfs):
    beam = _get_beam_from_model(model_tfs)

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

    return calibration_factors


def _get_dispersion_from_phase(normalised_dispersion, beta):
    # Compute the dispersion from phase
    d_phase = normalised_dispersion["amp"] * np.sqrt(beta["phase"])

    # And the the associated error
    d_phase_err = (normalised_dispersion["amp_err"] * np.sqrt(beta["phase_err"])) ** 2
    d_phase_err += (
        1/2
        * normalised_dispersion["amp"]
        / np.sqrt(beta["phase"])
        * beta["phase_err"]
    ) ** 2
    d_phase_err = np.sqrt(d_phase_err)

    return d_phase, d_phase_err


def _get_dispersion_fit(positions, dispersion_values, dispersion_err):
    def dispersion_function(x, a, b):
        return a * x + b

    # Filter the values we have to only keep the asked BPMs
    values = dispersion_values[dispersion_values.index.isin(positions.index)]
    err = dispersion_err[dispersion_values.index.isin(positions.index)]

    # Get the curve fit for the expected affine function
    fit, fit_cov = curve_fit(dispersion_function, positions, values, sigma=err)
    # FIXME add sigma=err

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


def _get_factors_from_dispersion(dispersion):
    # Get the ratios, those are our calibration factors
    factors = dispersion["phase"] / dispersion["amp"]

    # Code in BBs
    calibration_error = (dispersion["phase_err"] / dispersion["amp"]) ** 2
    calibration_error += (
        dispersion["amp_err"] * (dispersion["phase"] / (dispersion["amp"] ** 2))
    ) ** 2
    calibration_error = np.sqrt(calibration_error)

    return factors, calibration_error


def _get_factors_from_dispersion_fit(dispersion):
    # Get the ratios, those are our calibration factors
    factors = dispersion["phase_fit"] / dispersion["amp"]

    calibration_error = (dispersion["phase_fit_err"] / dispersion["amp"]) ** 2
    calibration_error += (
        dispersion["amp_err"] * dispersion["phase_fit"] / (dispersion["amp"] ** 2)
    ) ** 2
    calibration_error = np.sqrt(calibration_error)

    return factors, calibration_error


def _get_calibration_factors_dispersion(
    ip, plane, beta_phase_tfs, dispersion_tfs, model_tfs
):
    beam = _get_beam_from_model(model_tfs)

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
    return calibration_factors


def _write_calibration_tfs(calibration_factors, plane, method, output_path):
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
    # Load the tfs for beta from phase, beta from amp and from the model
    beta_phase_tfs = _get_beta_from_phase_tfs(opt.input_path)
    beta_amp_tfs = _get_beta_from_amp_tfs(opt.input_path)
    model_tfs = tfs.read(opt.model_path / MODEL_TFS, index=TFS_INDEX)

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
                    ip, plane, beta_phase_tfs[plane], beta_amp_tfs[plane], model_tfs
                )
            elif opt.method == "dispersion" and plane == "X":
                factors = _get_calibration_factors_dispersion(
                    ip, plane, beta_phase_tfs[plane], dispersion_tfs, model_tfs
                )

            c_factors[plane] = c_factors[plane].append(factors)

        # Write the TFS file to the desired output directory
        # There's no calibratuon factors on the Y plane for dispersion
        if opt.method == "dispersion" and plane != "X":
            continue
        _write_calibration_tfs(c_factors[plane], plane, opt.method, opt.output_path)

    return c_factors


if __name__ == "__main__":
    main()
