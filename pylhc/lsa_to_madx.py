"""
LSA to MAD-X
------------

This script is meant to convert various ``LSA`` knobs to their ``MAD-X`` equivalent scripts.

.. code-block:: none

    usage: lsa_to_madx.py [-h] --optics OPTICS [--knobs [KNOBS ...]] [--file FILE]

    LSA Knob to MAD-X Converter.This script can be given an LSA LHC optics, a list
    of LSA knobs or a file with LSA knobs and will, for each knob, retrieve the
    affected LHC power circuits and determine the corresponding MAD-X variables
    changes. It will then output both definition files and MAD-X scripts reproducing
    the provided knobs.

    options:
    -h, --help           show this help message and exit
    --optics OPTICS      The LSA name of the optics for which the knobs are defined.
    --knobs [KNOBS ...]  The full LSA names of any knob to convert to their MAD-X equivalent.
    --file FILE          Name of a file with knob names and strength factors to use. A
                         single knob should be written per line, and lines starting with a
                         '#' character will be ignored.

One can simply want to find out the ``MAD-X`` way to reproduce a given knob.
For instance, to find reproduce ``LHCBEAM/MD_ATS_2022_05_04_B1_RigidWaitsShift_IP1pos``, which is defined in the ``R2022a_A30cmC30cmA10mL200cm`` optics one would run:

.. code-block:: none

    python -m pylhc.lsa_to_madx \\
        --optics R2022a_A30cmC30cmA10mL200cm \\
        --knobs LHCBEAM/MD_ATS_2022_05_04_B1_RigidWaitsShift_IP1pos

Two files, **LHCBEAM_MD_ATS_2022_05_04_B1_RigidWaitsShift_IP1pos_definition.tfs** and **LHCBEAM_MD_ATS_2022_05_04_B1_RigidWaitsShift_IP1pos_knob.madx** will be written to disk.

.. warning::
    In ``MAD-X``, variable names with 48 or more characters will cause an issue.
    As a consequence, this script will automatically truncate the knob name if needed when created the trim variable name.
    One should not be surprised if long ``LSA`` knob names appear slightly differently in the created ``MAD-X`` files, then functionality stays intact.

    For instance, the knob ``LHCBEAM/MD_ATS_2022_05_04_B1_RigidWaitsShift_IP1pos`` will lead to the following trim variable definition:
    
    .. code-block:: fortran

        trim_D_ATS_2022_05_04_B1_RigidWaitsShift_IP1pos = 1.0;

In order to reproduce a specific machine configuration at a given time, one can gather all knobs and their trim values for this configuration in a text file and feed this file to the script.
In this file, each line should hold a knob name as it appears in LSA and its trim value.
Lines starting with a ``#`` character will be ignored.

For instance the following **knobs.txt** file gathers a 2017 LHC configuration:

.. code-block:: none

    # Nonlinear Corrections
    LHCBEAM/2017_IRNL_IR1a4	0.5
    LHCBEAM/2017_IRNL_IR1b3_couplFD	1.0
    LHCBEAM/2017_IRNL_a3b3_tuneFD 1.0
    LHCBEAM/2017_IRNL_b4 1.0
    LHCBEAM/2017_NL_IR1_30cm_kcssxr1 1.0
    LHCBEAM/2018_IRNL_IR1a3 1.0
    LHCBEAM/2018_IRNL_IR1b3 1.0
    LHCBEAM/2018_IRNL_IR5a3 1.0
    LHCBEAM/IR5_a4_2018 1.0
    # Local correctors close to IP
    LHCBEAM/2017_ATS_Inj_LocalCoupling 1.0
    LHCBEAM/2017_ATS_LocalCorrection 1.0
    # Chromatic Coupling
    LHCBEAM/chromatic_coupling_b1_40cm_ctpps2_2017_v2 -1.0
    LHCBEAM/chromatic_coupling_b2_40cm_ctpps2_2017 -1.0
    # Beta-beat
    LHCBEAM/b1_global_beta_beating_40cm_ctpps_2017_v2 1.0
    LHCBEAM/b2_global_beta_beating_40cm_ctpps_2017 1.0
    LHCBEAM/2017_beam1_beta_CrossingAngleCompensation 1.0
    LHCBEAM/2017_30cm_GlobalCorr_wXing_Beam2_v2 1.0
    LHCBEAM/B2_2017_global_correction_Xing 1.0
    # Total Phase advance
    LHCBEAM/2017_B2_MQT_total_phase_correction 1.0
    LHCBEAM/B1_2017_MQT_total_phase_correction 1.0
    # Arc-by-arc coupling
    LHCBEAM/global_coup_knob_all_correctors	-1.0

One can provide this file with the corresponding optics and get a ``MAD-X`` file for each defined knob with the appropriate trim value.
In this case many files will be created.
The call would be:

.. code-block:: bash

    python -m pylhc.lsa_to_madx \\
        --optics R2017aT_A30C30A10mL300_CTPPS2 \\
        --file knobs.txt

Hint: the knobs active at a given time can be retrieved with the `~pylhc.machine_settings_info` script. 
"""
import argparse
import re
import string

from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd
import tfs

from omc3.utils import logging_tools
from omc3.utils.contexts import timeit

from pylhc.data_extract.lsa import LSAClient

LOG = logging_tools.get_logger(__name__)
ALLOWED_IN_MADX_NAMES = "_" + string.ascii_letters + string.digits

# ----- Helper functions ----- #


def parse_knobs_and_trim_values_from_file(knobs_file: Path) -> Dict[str, float]:
    """
    Parses a file for LSA knobs and their trim values. Each line should be a knob name
    following by a number of the trim value. If no value is written, it defaults
    to ``1.0``. Lines starting with a ``#`` are ignored.

    Args:
        knobs_file (~pathlib.Path): Path to the file with definitions.

    Returns:
        A `dict` with as keys the parsed knob names and as values their associated trims.
    """
    knob_lines = [line for line in Path(knobs_file).read_text().splitlines() if not line.startswith("#")]
    results = {}

    for line in knob_lines:
        knob = line.split()[0]  # this is the knob name
        try:  # catch the trim value, which is a number after the knob name
            trim = float(line.split()[1])
            results[knob] = trim
        except IndexError:  # there is no number, we default the trim value to 1.0
            LOG.debug(f"No trim value was specified for knob '{knob}' - defaulting to 1.0")
            results[knob] = 1.0
    return results


def get_sign_madx_vs_lsa(madx_name: str) -> int:
    """
    Return the sign convention between madx and lsa for the given variable.

    Args:
        madx_name (str): Name of the variable as used in MAD-X.

    Returns:
        int: 1 or -1.

    """
    # Test for skewness:
    match = re.match(r"^.c?[qsodt]s", madx_name)  # k, then maybe a "c", then type, then skew
    if match is not None:
        LOG.debug(f"{madx_name} belongs to a skew magnet: Sign is inverted betwen LSA and MAD-X")
        return -1

    # Test for Q2
    if madx_name.lower().startswith("ktqx2"):
        LOG.debug(f"{madx_name} belongs to Q2: Sign convention is unknown")
        raise NotImplementedError(
            "Q2 is not implemented yet, as the sign convention LSA-MADX is not known. "
            "Please check and implement."
        )

    LOG.debug(f"Sign of {madx_name} is the same in LSA and MAD-X.")
    return 1


def get_madx_script_from_definition_dataframe(deltas_df: tfs.TfsDataFrame, lsa_knob: str, trim: float = 1.0,
                                              by_reference: bool = True, verbose: bool = False
                                              ) -> str:
    """
    Given the extracted definition dataframe of an LSA knob - as returned by
    `~pylhc.data_extract.lsa.LSAClient.get_knob_circuits` - this function will generate the
    corresponding ``MAD-X`` text commands that would reproduce this knob in a script.

    Args:
        deltas_df (~tfs.frame.TfsDataFrame): The extracted definition dataframe of an ``LSA`` knob. This
            can be obtained with `~pylhc.data_extract.lsa.LSAClient.get_knob_circuits`.
        lsa_knob (str): The complete ``LSA`` name of the knob, including any ``LHCBEAM[12]?/`` part.
        trim (float): The trim value to write for the knob. Defaults to 1.
        by_reference (bool): If true, creates an _init variable and uses deferred expressions (`:=`)
                            so that changing the trim later in the script changes all
                            variables as well. Might be problematic if a variable depends on
                            multiple trim-knobs.
                            Otherwise `variable = variable + delta*trim` is used.
        verbose (bool): Adds some extra comments into the madx code.

    Returns:
        The complete ``MAD-X`` script to reproduce the knob, as a string.
    """
    LOG.debug(f"Determining MAD-X commands to reproduce knob '{lsa_knob}'")
    change_commands = [f"! Start of change commands for knob: {lsa_knob}"]

    # Set this to 1 by default but can be changed by the user to reproduce a given trim
    trim_variable = _get_trim_variable(lsa_knob)
    if verbose:
        change_commands.append("! Change this value to reproduce a different trim")
        change_commands.append("! Beware some knobs are not so linear in their trims")
    change_commands.append(f"{trim_variable} = {trim};")
    change_commands.append("! Impacted variables:")

    deltas = _get_delta(deltas_df)

    # write all inits first (looks nicer in madx)
    if by_reference:
        for variable in deltas.keys():
            variable_init = f"{variable}_init"
            change_commands.append(f"{variable_init:<17} = {variable};")

    # write knob-definition
    for variable, delta in deltas.items():
        delta = get_sign_madx_vs_lsa(variable) * delta
        # Parenthesis around the delta below are important for MAD-X to not
        # mess up parsing of "var = var + -value" if delta_k is negative
        if by_reference:
            variable_init = f"{variable}_init"
            change_commands.append(f"{variable:<12} := {variable_init:^19} + ({delta:^25}) * {trim_variable};")
        else:
            change_commands.append(f"{variable:<12} = {variable:^15} + ({delta:^25}) * {trim_variable};")

    change_commands.append(f"! End of change commands for knob: {lsa_knob}\n")
    return "\n".join(change_commands)


def _get_trim_variable(lsa_knob: str) -> str:
    """
    Generates the ``MAD-X`` trim variable name from an ``LSA`` knob.
    Handles the variable name character limit of ``MAD-X``.
    """
    # Handle the LHCBEAM-Part:
    lhcbeam = re.match(r"^LHCBEAM(\d?)", lsa_knob.upper())
    if lhcbeam is not None:
        madx_knob = lsa_knob.split("/")[-1]  # without the LHCBEAM[12]?/ part
        if lhcbeam.group(1):
            # beam has a number: there might be two knobs with this name
            madx_knob = f"B{lhcbeam.group(1)}_{madx_knob}"
    else:
        madx_knob = lsa_knob

    # Clean up the name
    madx_knob = madx_knob.replace("-", "_").replace("/", "_")
    for c in set(madx_knob):
        if c not in ALLOWED_IN_MADX_NAMES:
            LOG.debug(f"Knob contains the character '{c}', which is not allowed in madx. Removing.")
            madx_knob = madx_knob.replace(c, "")

    # MAD-X will crash if the variable name is >48 characters or longer! It will also silently fail
    # if the variable name starts with an underscore or a digit. Adding "trim_" at the start circumvents
    # the latter two, and we make sure to truncate the knob so that the result is <=47 characters
    if len(madx_knob) > 42:
        madx_knob = madx_knob[-42:]
        LOG.debug(f"Truncated knob name to '{madx_knob}'.")

    trim_variable = f"trim_{madx_knob.lstrip('_')}"
    LOG.debug(f"Converted knob-name '{lsa_knob}' to trim variable '{trim_variable}'.")
    return trim_variable


def _get_delta(deltas_df: tfs.TfsDataFrame) -> pd.Series:
    """ Get the correct delta-column """
    if "DELTA_K" not in deltas_df.columns:
        LOG.debug("Using DELTA_KL column.")
        return deltas_df.DELTA_KL

    if "DELTA_KL" not in deltas_df.columns:
        LOG.debug("Using DELTA_K column.")
        return deltas_df.DELTA_K

    if (deltas_df.DELTA_K.astype(bool) & deltas_df.DELTA_KL.astype(bool)).any():
        raise ValueError("Some entries of DELTA_KL and DELTA_K seem to both be given. "
                         "This looks like a bug. Please investigate.")

    LOG.debug("Both DELTA_K and DELTA_KL columns present, merging columns.")
    return pd.Series(np.where(deltas_df.DELTA_K.astype(bool), deltas_df.DELTA_K, deltas_df.DELTA_KL),
                     index=deltas_df.index)


# ----- Script Part ----- #


def _get_args():
    """Parse Commandline Arguments."""
    parser = argparse.ArgumentParser(
        description="LSA Knob to MAD-X Converter."
        "This script can be given an LSA LHC optics, a list of LSA knobs or a file with LSA knobs "
        "and will, for each knob, retrieve the affected LHC power circuits and determine the "
        "corresponding MAD-X variables changes. It will then output both definition files and MAD-X "
        "scripts reproducing the provided knobs."
    )
    parser.add_argument(
        "--optics", dest="optics", type=str, required=True, help="The LSA name of the optics for which the knobs are defined."
    )
    parser.add_argument(
        "--knobs",
        dest="knobs",
        type=str,
        nargs="*",
        required=False,
        help="The full LSA names of any knob to convert to their MAD-X equivalent.",
    )
    parser.add_argument(
        "--file",
        dest="file",
        type=str,
        required=False,
        help="Name of a file with knob names and strength factors to use. "
        "A single knob should be written per line, and lines starting with a # "
        "character will be ignored.",
    )
    return parser.parse_args()


def main():
    options = _get_args()
    lsa_optics = options.optics

    if not options.knobs and not options.file:
        LOG.error("Need to provide either a list of knob names or a file with 1 knob name per line")
        exit(1)

    if options.knobs and options.file:
        LOG.error("Either provide knobs at the command line or from a text file, but not both")
        exit(1)

    if options.file and Path(options.file).is_file():
        LOG.info(f"Loading knob names from file '{options.file}'")
        knobs_dict = parse_knobs_and_trim_values_from_file(Path(options.file))
    else:  # given at the command line with --knobs, we initialise trim values to 1
        knobs_dict = {knob: 1.0 for knob in options.knobs}

    LOG.info("Instantiating LSA client")
    lsa_client = LSAClient()
    unfound_knobs = []

    with timeit(lambda elapsed: LOG.info(f"Processed all given knobs in {elapsed:.2f}s")):
        for lsa_knob, trim_value in knobs_dict.items():
            LOG.info(f"Processing LSA knob '{lsa_knob}'")
            try:  # next line might raise if knob not defined for the given optics
                knob_definition = lsa_client.get_knob_circuits(knob_name=lsa_knob, optics=lsa_optics)
                madx_commands_string = get_madx_script_from_definition_dataframe(
                    deltas_df=knob_definition, lsa_knob=lsa_knob, trim=trim_value
                )

            except (OSError, IOError):  # raised by pjlsa if knob not found
                LOG.warning(f"Could not find knob '{lsa_knob}' in the given optics '{lsa_optics}' - skipping")
                unfound_knobs.append(lsa_knob)

            else:  # we've managed to find knobs
                # Don't write this in the try block, as it could raise IOError when failing
                definition_file = f"{lsa_knob.replace('/', '_')}_definition.tfs"
                LOG.debug(f"Writing knob definition TFS file at '{definition_file}'")
                tfs.write(definition_file, knob_definition, save_index=True)

                madx_file = f"{lsa_knob.replace('/', '_')}_knob.madx"
                LOG.debug(f"Writing MAD-X commands to file '{madx_file}'")
                Path(madx_file).write_text(madx_commands_string)

    if unfound_knobs:
        LOG.info(f"The following knobs could not be found in the '{lsa_optics}' optics: \n\t\t" + "\n\t\t".join(unfound_knobs))


if __name__ == "__main__":
    main()
