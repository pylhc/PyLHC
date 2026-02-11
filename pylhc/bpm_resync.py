"""
BPM Synchronization
-------------------

This script resyncs the BPMs from the `LER` and `HER` rings of `SuperKEKB`.
Those BPMs are often not aligned time-wise and the values can be off by a few turns.
The resynchronization is done by looking up the phase advance of each BPM to retrieve the turn
offset.
This requires a frequency and an optics analysis of the unsynchronized turn by turn data.

The script takes as input the original turn by turn data, the optics directory containing the
results of the optics analysis, as well as the output filename where the turn by turn data will be
written, in ASCII SDDS format.


Arguments:

*--Required--*

- **input** *(Path,TbtData)*:

    Input turn by turn data to be resynchronized.
    Can take the form of a `Path` to a file or directly a `TbtData` object.

    flags: **['--input']**

- **optics_dir** *(Path)*:

    Optics path, must contain the `total_phase_{x,y}.tfs` files.

    flags: **['--optics_dir']**

- **output_file** *(Path)*:

    Output file path where to write the turn by turn data.
    The directory will be created if necessary.

    flags: **['--output_file']**

- **ring** *(str)*:

    Ring name, either `LER` or `HER`.

    flags: **['--ring']**

    choices: ``('LER', 'HER')``

*--Optional--*

- **tbt_datatype** *(str)*:
    Datatype of the TurnByTurn data provided as input.

    flags: **['--tbt_datatype']**

    choices: list of datatypes supported by `turn_by_turn`, in `turn_by_turn.io.TBT_MODULES`

    default: ``lhc``
"""

from copy import deepcopy
from pathlib import Path

import numpy as np
import tfs
import turn_by_turn as tbt
from generic_parser import EntryPointParameters, entrypoint
from generic_parser.dict_parser import ArgumentError
from omc3.utils import logging_tools

from pylhc.constants.bpm_resync import DEFAULT_DATATYPE, PHASE_FILE, RINGS

LOGGER = logging_tools.get_logger(__name__)


def _get_params() -> dict:
    """
    Parse Commandline Arguments and return them as options.

    Returns:
        dict
    """

    return EntryPointParameters(
        input={
            "required": True,
            "help": "Input turn by turn data to be resynchronized. Can take the form of a `Path` to"
            "a file or directly a `TbtData` object.",
        },
        optics_dir={
            "type": Path,
            "required": False,
            "help": "Optics path, must contain the `total_phase_{x,y}.tfs` files.",
        },
        output_file={
            "type": Path,
            "required": False,
            "help": "Output file path where to write the turn by turn data. The directory will be"
            "created if necessary.",
        },
        ring={
            "type": str,
            "required": True,
            "choices": RINGS,
            "help": (f"Ring name, from {RINGS}"),
        },
        tbt_datatype={
            "type": str,
            "required": False,
            "choices": list(tbt.io.TBT_MODULES.keys()),
            "default": DEFAULT_DATATYPE,
            "help": "Datatype of the TurnByTurn data",
        },
    )


def sync_tbt(original_tbt, phase_dataframe_fmt, ring):
    """Resynchronize the BPMS in the the turn by turn data based on the phase advance."""
    # Copy the original turn by turn data
    synced_tbt = deepcopy(original_tbt)

    # HER and LER are in opposite direction for the phase
    ring_dir = 1 if ring == "HER" else -1

    # Some BPMs can exist in a plane but not the other, we need to check both planes to be sure
    already_processed = set()
    for axis in ("x", "y"):
        phase_df = tfs.read(str(phase_dataframe_fmt).format(axis=axis))
        bpms = phase_df["NAME"]
        qx = phase_df.headers["Q1"]
        qy = phase_df.headers["Q2"]
        dphase = phase_df[f"DELTAPHASE{axis.upper()}"]
        tune = (1 - qx) if axis == "x" else (1 - qy)

        # Iterate through all the BPMs and check their phase advance
        for i, bpm in enumerate(bpms):
            # Check if we've seen that BPM before in the other plane
            if bpm in already_processed:
                continue
            already_processed.add(bpm)

            # The phase advance divided by the tune will tell us how off the BPM is
            ntune = dphase[i] / tune
            abs_n = abs(ntune)

            # If the difference is close to 1, that's one turn
            # Otherwise, it's likely -2 turns
            if abs_n >= 0.8:
                mag = 1
            elif abs_n >= 0.1:
                mag = -2
            else:
                mag = 0

            # The final number of turns also depends on the sign of the phase
            final_correction = int(mag * np.sign(ntune) * ring_dir)

            if final_correction != 0:
                LOGGER.info(
                    f"  {bpm:15s} -> turn correction of {final_correction} (ntune={ntune:.2f})"
                )

            # Shift the data
            for plane in ("X", "Y"):
                matrix = synced_tbt.matrices[0].__getattribute__(plane)
                orig_row = original_tbt.matrices[0].__getattribute__(plane).loc[bpm]
                matrix.loc[bpm] = orig_row.shift(final_correction, fill_value=0)

    return synced_tbt


@entrypoint(_get_params(), strict=True)
def main(opt):
    # Set the phase path
    phase_dataframe_fmt = opt.optics_dir / PHASE_FILE

    # Open the TbT file if needed
    if isinstance(opt.input, Path):
        original_tbt = tbt.read(opt.input, datatype=opt.tbt_datatype)
    elif isinstance(opt.input, tbt.TbtData):
        original_tbt = opt.input
    else:
        raise ArgumentError("input must be either a Path or a TbtData object")

    # Synchronise TbT
    LOGGER.info(f"Resynchronizing {opt.optics_dir.name}...")
    synced_tbt = sync_tbt(original_tbt, phase_dataframe_fmt, opt.ring)  # type: ignore

    # Save the resynced turn by turn data
    opt.output_file.parent.mkdir(exist_ok=True)
    tbt.write(opt.output_file, synced_tbt)


if __name__ == "__main__":
    main()
