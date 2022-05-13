"""
LSA to MAD-X
------------

This script is meant to convert various LSA knobs to their MAD-X equivalent.

.. code-block:: none

    usage: python -m pylhc.lsa_to_madx [-h] --optics OPTICS [--knobs [KNOBS ...]] [--file FILE]

    LSA Knob to MAD-X Converter.This script can be given an LSA LHC optics,
    a list of LSA knobs or a file with LSA knobs and will, for each knob,
    determine  the affected LHC power circuits and the corresponding ``MAD-X``
    variables; then output files with the corresponding ``add2expr`` commands.

    optional arguments:
    -h, --help           show this help message and exit
    --optics OPTICS      The LSA name of the optics for which the knobs are defined.
    --knobs [KNOBS ...]  The various knob names to convert to their MAD-X equivalent.
    --file FILE          Name of a file with knob names and strength factors to use. A single 
                        knob should be written per line, and lines starting with a ``#`` character 
                        will be ignored.
"""
import argparse

from pathlib import Path
from typing import Dict

from omc3.utils import logging_tools
from omc3.utils.contexts import timeit
from omc3.utils.mock import cern_network_import

LOG = logging_tools.get_logger(__name__)
pjlsa = cern_network_import("pjlsa")

# ----- Helper functions ----- #


def get_power_circuits_and_factors_from_lsa_knob(lsa_client: pjlsa.LSAClient, optics: str, lsa_knob: str) -> Dict[str, float]:
    """
    Given the name of a knob defined in ``LSA`` (see for instance **LSA Application Management App**),
    will find all the various power circuits present in the knob and the factor corresponding to each.

    Args:
        lsa_client (pjlsa.LSAClient): an instantiated ``LSAClient`` to use to connect to ``LSA``.
        optics (str): the name of the optics for which the knob is defined.
        lsa_knob (str): the name of the knob as defined in the ``LSA`` database.

    Returns:
        A dictionary with the power circuits as keys and the corresponding factors as values.
    """
    LOG.debug(f"Getting LSA power circuits and factors defined in LSA knob '{lsa_knob}'")
    return lsa_client.getKnobFactors(lsa_knob, optics)


def lsa_power_circuit_to_madx_variable(lsa_client: pjlsa.LSAClient, lsa_power_circuit: str) -> str:
    """
    Given the name of a power circuit defined in ``LSA``, will find the corresponding ``MAD-X`` variable
    in the LHC sequence or opticsfile.

    Args:
        lsa_client (pjlsa.LSAClient): an instantiated ``LSAClient`` to use to connect to ``LSA``.
        lsa_power_circuit (str): the name of the power circuit in ``LSA``.

    Returns:
        The ``MAD-X`` variable corresponding to the given ``LSA`` power circuit.
    """
    LOG.debug(f"Finding MAD-X name for LSA circuit '{lsa_power_circuit}'")
    power_circuit = lsa_power_circuit.split("/")[0]  # this is the knob name without the /K[1-9][SL] part
    return lsa_client.findMadStrengthNameByPCName(power_circuit)


def process_lsa_knob_to_madx_commands(lsa_client: pjlsa.LSAClient, optics: str, lsa_knob: str, madx_knob: str) -> str:
    """
    Given an LSA knob and the optics in which it's applied, finds the corresponding MAD-X variables
    to change and returns the exact ``add2expr`` command to do so.

    Args:
        lsa_client (pjlsa.LSAClient): an instantiated ``LSAClient`` to use to connect to ``LSA``.
        optics (str): the ``LSA`` name of the optics for which the knob is defined.
        lsa_knob (str): the name of the knob as defined in the ``LSA`` database.
        madx_knob (str): the name of the``MAD-X`` knob corresponding to the LSA knob.

    Returns:
        A full string with all of the ``add2expr`` commands to be executed in ``MAD-X``.
    """
    LOG.debug(f"Determining MAD-X expressions for LSA knob '{lsa_knob}'")
    change_commands = [f"! Start of change commands for knob: {lsa_knob}"]
    try:
        power_circuit_factors = get_power_circuits_and_factors_from_lsa_knob(lsa_client, optics, lsa_knob)
        for power_circuit, factor in power_circuit_factors.items():  # determine madx variable and expression change
            madx_variable = lsa_power_circuit_to_madx_variable(lsa_client, power_circuit)
            change_commands.append(f"add2expr,var={madx_variable},expr={factor}*{madx_knob};")
        change_commands.append(f"! End of change commands for knob: {lsa_knob}\n")
        return "\n".join(change_commands)
    except:  # In case it's not possible to find the MAD-X variable?
        return ""


# ----- Script Part ----- #


def _get_args():
    """Parse Commandline Arguments."""
    parser = argparse.ArgumentParser(
        description="LSA Knob to MAD-X Converter."
        "This script can be given an LSA LHC optics, a list of LSA knobs or a file with LSA knobs "
        "and will, for each knob, determine the affected LHC power circuits and the corresponding "
        "MAD-X variables; then output files with the corresponding ``add2expr`` commands."
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
        help="The various knob names to convert to their MAD-X equivalent.",
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


if __name__ == "__main__":
    options = _get_args()
    lsa_optics = options.optics

    if not options.knobs and not options.file:
        LOG.error("Need to provide either a list of knob names or a file with 1 knob name per line")
        exit(1)

    if options.knobs and options.file:
        LOG.error(f"Either provide knobs at the command line or from a text file, but not both")
        exit(1)

    if options.file and Path(options.file).is_file():
        LOG.info(f"Loading knob names from file '{options.file}'")
        knobs = [line for line in Path(options.file).read_text().splitlines() if not line.startswith("#")]
    else:
        knobs = options.knobs

    LOG.info("Instantiating LSA client")
    lsa_client = pjlsa.LSAClient()

    with timeit(lambda elapsed: LOG.info(f"Processed all knobs in {elapsed:.2f}s")):
        for lsa_knob in knobs:
            LOG.info(f"Processing LSA knob '{lsa_knob}'")
            mad_name = lsa_knob.split("/")[1]
            madx_commands_string = process_lsa_knob_to_madx_commands(
                lsa_client=lsa_client, optics=lsa_optics, lsa_knob=lsa_knob, madx_knob=mad_name
            )
            LOG.debug(f"Writing MAD-X commands to file '{mad_name}_knob.madx'")
            Path(f"{mad_name}_knob.madx").write_text(madx_commands_string)
