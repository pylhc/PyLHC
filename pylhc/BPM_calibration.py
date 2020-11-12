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
    EXT
)
from pylhc.constants.calibration import (
    BPMS,
    BETA_STAR_ESTIMATION,
    CALIBRATION_NAME,
    IPS,
    LABELS,
    METHODS,
    MODEL_TFS,
    TFS_INDEX
)
from pylhc.constants.general import (
    BEAMS,
    PLANES
)
import tfs


LOG = logging_tools.get_logger(__name__)


def _get_params() -> dict:
    """ Parse Commandline Arguments and return them as options. """

    return EntryPointParameters(
        input_path=dict(
            flags=["--input", "-i"],
            required=True,
            type=Path,
            help="Measurements path."),
        model_path=dict(
            flags=["--model", "-m"],
            type=Path,
            required=True,
            help="Model path associated to the measurements."),
        output_path=dict(
            flags=["--outputdir", "-o"],
            type=Path,
            required=True,
            help="Output directory where to write the calibration factors."),
        method=dict(
            flags=["--method"],
            type=str,
            required=False,
            choices=METHODS,
            default=METHODS[0],
            help=("Method to be used to compute the calibration factors. "
                  "The Beta function is used by default.")),
    )


def _get_beta(beta, input_path):
    tfs_files = dict()
    for plane in PLANES:
        beta_file = f'{beta}{plane.lower()}{EXT}'
        expected_path = input_path / beta_file

        if not expected_path.is_file():
            msg = f"File {beta_file} couldn't be found in directory {input_path}"
            LOG.error(msg)
            raise FileNotFoundError(msg)

        tfs_files[plane] = tfs.read(expected_path, index=TFS_INDEX)
    return tfs_files


def _get_beta_from_phase(input_path):
    return _get_beta(BETA_NAME, input_path)


def _get_beta_from_amp(input_path):
    return _get_beta(AMP_BETA_NAME, input_path)


def _get_beam_from_model(model_tfs):
    beam = int(model_tfs.SEQUENCE[-1])
    if beam not in BEAMS:
        msg = f"Could not find a correct value for beam in model: {beam}"
        LOG.error(msg)
        raise ValueError(msg)
    return beam


def _get_beta_fit(positions, beta_values, beta_err):
    def phase_function(x, a, b):
        return a + ((x - b) ** 2) / a

    # Get the rough IP position and beta star for the initial values
    ip_position = (positions[-1] - positions[0]) / 2
    initial_values = (BETA_STAR_ESTIMATION, ip_position)

    # Get the curve fit for the expected 1parabola
    fit, fit_cov = curve_fit(phase_function,
                             positions,
                             beta_values,
                             p0=initial_values,
                             sigma=beta_err,
                             maxfev=1000000)

    # Get the error from the covariance matrix
    fit_err = np.sqrt(np.diag(fit_cov))

    # Get the fitted beta and add the errors to get min/max values
    beta_fit = phase_function(positions, fit[0], fit[1])
    beta_max_fit = phase_function(positions,
                                  fit[0] + fit_err[0],
                                  fit[1] + fit_err[1])
    beta_min_fit = phase_function(positions,
                                  fit[0] - fit_err[0],
                                  fit[1] - fit_err[1])
    beta_star_fit_err = (beta_max_fit - beta_min_fit) / 2

    return beta_fit, beta_star_fit_err


def _get_factors_from_phase(beta_phase, beta_amp, beta_phase_err, beta_amp_err):
    # Get the ratios, those are our calibration factors
    ratio_phase_amplitude = np.sqrt(beta_phase / beta_amp)

    # Now compute the errors
    calibration_error = (beta_phase_err ** 2) / (4 * beta_amp * beta_phase)
    calibration_error += (beta_phase * (beta_amp_err ** 2)) / (4 * (beta_amp ** 3))
    calibration_error = np.sqrt(calibration_error)

    return ratio_phase_amplitude, calibration_error


def _get_factors_from_phase_fit(beta_phase_fit, beta_amp, beta_phase_fit_err, beta_amp_err):
    # Get the ratios, those are our calibration factors
    ratio_phasefit_amplitude = np.sqrt(beta_phase_fit / beta_amp)

    # Now compute the errors
    calibration_error = (beta_phase_fit_err ** 2) / (4 * beta_amp * beta_phase_fit)
    calibration_error += (beta_phase_fit * (beta_amp_err ** 2)) / (4 * (beta_amp ** 3))
    calibration_error = np.sqrt(calibration_error)

    return ratio_phasefit_amplitude, calibration_error


def _get_calibration_factors_beta(ip, plane, beta_phase_tfs, beta_amp_tfs, model_tfs):
    beam = _get_beam_from_model(model_tfs)

    # Filter our TFS files to only keep the BPMs for the selected IR
    bpms = beta_phase_tfs.reindex(BPMS[ip][beam]).dropna().index

    # Get the positions and the beta values for those BPMs
    positions = beta_phase_tfs.loc[bpms, 'S'].dropna()
    beta_phase = beta_phase_tfs.loc[bpms, f'{BETA}{plane}'].dropna()
    beta_phase_err = beta_phase_tfs.loc[bpms, f'{ERR}{BETA}{plane}'].dropna()
    beta_amp = beta_amp_tfs.loc[bpms, f'{BETA}{plane}'].dropna()
    beta_amp_err = beta_amp_tfs.loc[bpms, f'{ERR}{BETA}{plane}'].dropna()

    # Curve fit the beta from phase values
    beta_phase_fit, beta_phase_fit_err = _get_beta_fit(positions,
                                                       beta_phase,
                                                       beta_phase_err)

    # Get the calibration factors for each method: from phase and phase fit
    calibration_phase, calibration_phase_err = _get_factors_from_phase(
                                                    beta_phase,
                                                    beta_amp,
                                                    beta_phase_err,
                                                    beta_amp_err)
    calibration_phase_fit, calibration_phase_fit_err = _get_factors_from_phase_fit(
                                                            beta_phase_fit,
                                                            beta_amp,
                                                            beta_phase_fit_err,
                                                            beta_amp_err)

    # Assemble the calibration factors in one dataframe
    calibration_factors = pd.concat([positions,
                                     calibration_phase,
                                     calibration_phase_err,
                                     calibration_phase_fit,
                                     calibration_phase_fit_err], axis=1)
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
    tfs_name = f'{CALIBRATION_NAME[method]}{plane.lower()}{EXT}'
    file_path = output_path / tfs_name
    tfs.write_tfs(file_path, calibration_factors, save_index=False)


@entrypoint(_get_params(), strict=True)
def main(opt):
    # Load the tfs for beta from phase, beta from amp and from the model
    beta_phase_tfs = _get_beta_from_phase(opt.input_path)
    beta_amp_tfs = _get_beta_from_amp(opt.input_path)
    model_tfs = tfs.read(opt.model_path / MODEL_TFS, index=TFS_INDEX)

    # Compute the calibration factors and their errors for each plane
    c_factors = dict()
    for plane in PLANES:
        c_factors[plane] = tfs.TfsDataFrame()
        for ip in IPS:
            if opt.method == 'beta':
                factors = _get_calibration_factors_beta(ip,
                                                        plane,
                                                        beta_phase_tfs[plane],
                                                        beta_amp_tfs[plane],
                                                        model_tfs)
            elif opt.method == 'dispersion':
                factors = 'lol fix me plz'

            c_factors[plane] = c_factors[plane].append(factors)

        # Write the TFS file to the desired output directory
        _write_calibration_tfs(c_factors[plane], 
                               plane, 
                               opt.method, 
                               opt.output_path)

    return c_factors


if __name__ == "__main__":
    main()
