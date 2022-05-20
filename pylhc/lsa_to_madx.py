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

import tfs

from omc3.utils import logging_tools
from omc3.utils.contexts import timeit

from pylhc.data_extract.lsa import LSA

LOG = logging_tools.get_logger(__name__)

# ----- Helper functions ----- #


def get_madx_script_from_deltas_dataframe(df: tfs.TfsDataFrame, knob: str) -> str:
    """"""
    LOG.debug(f"Determining MAD-X commands to reproduce knob '{knob}'")
    change_commands = [f"! Start of change commands for knob: {knob}"]

    # Set this to 1 by default but can be changed by the user to reproduce a given trim
    knob_itself = knob.split("/")[1]  # without the LHCBEAM/ part
    trim_variable = f"{knob_itself}_trim"
    change_commands.append("! Change this value to reproduce a different trim")
    change_commands.append(f"! Beware some knobs are not so linear in their trims")
    change_commands.append(f"{trim_variable} = 1;")

    for variable, delta_k in df.DELTA_K.items():
        change_commands.append(f"{variable} = {variable} + {delta_k} * {trim_variable};")
    change_commands.append(f"! End of change commands for knob: {knob}\n")
    return "\n".join(change_commands)


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
