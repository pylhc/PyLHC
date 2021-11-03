"""
Print Machine Settings Overview
-------------------------------

Prints an overview over the machine settings at a provided given time, or the current settings if
no time is given.

Can be run from command line, parameters as given in :meth:`pylhc.machine_settings_info.get_info`.

.. code-block:: none

    usage: machine_settings_info.py [-h] [--time TIME] [--knobs KNOBS [KNOBS ...]]
                                [--accel ACCEL] [--output_dir OUTPUT_DIR]
                                [--knob_definitions] [--source SOURCE] [--log]

    optional arguments:
      -h, --help            show this help message and exit
      --time TIME           UTC Time as 'Y-m-d H:M:S.f' format.
      --knobs KNOBS [KNOBS ...]
                            List of knobnames.
      --accel ACCEL         Accelerator name.
      --output_dir OUTPUT_DIR
                            Output directory.
      --source SOURCE       Source to extract data from.
      --knob_definitions    Set to extract knob definitions.
      --log                 Write summary into log (automatically done if no
                            output path is given).


:author: jdilly
"""
from collections import OrderedDict

import tfs
from generic_parser import DotDict, EntryPointParameters, entrypoint
from omc3.utils import logging_tools
from omc3.utils.iotools import PathOrStr
from omc3.utils.time_tools import AccDatetime, AcceleratorDatetime
from pathlib import Path
from typing import Tuple, Iterable, Dict

from pylhc.constants import machine_settings_info as const
from pylhc.data_extract.lsa import COL_NAME as LSA_COLUMN_NAME, LSA

LOG = logging_tools.get_logger(__name__)


# Main #########################################################################

def _get_params() -> dict:
    """Parse Commandline Arguments and return them as options."""
    return EntryPointParameters(
        time=dict(
            default=None,
            type=str,
            help="UTC Time as 'Y-m-d H:M:S.f' format."),
        knobs=dict(
            default=None,
            nargs="+",
            type=str,
            help="List of knobnames."),
        accel=dict(
            default='lhc',
            type=str,
            help="Accelerator name."),
        output_dir=dict(
            default=None,
            type=PathOrStr,
            help="Output directory."),
        knob_definitions=dict(
            action="store_true",
            help="Set to extract knob definitions."),
        source=dict(
            type=str,
            default="nxcals",
            help="Source to extract data from."),
        log=dict(
            action="store_true",
            help="Write summary into log (automatically done if no output path is given)."),
    )


@entrypoint(_get_params(), strict=True)
def get_info(opt) -> Tuple[AccDatetime, DotDict, DotDict, dict, dict]:
    """
    Get info about **Beamprocess**, **Optics** and **Knobs** at given time.

    Keyword Args:

    *--Optional--*

    - **accel** *(str)*:

        Accelerator name.

        default: ``lhc``


    - **knob_definitions**:

        Set to extract knob definitions.

        action: ``store_true``


    - **knobs** *(str)*:

        List of knobnames.

        default: ``None``


    - **log**:

        Write summary into log (automatically done if no output path is
        given).

        action: ``store_true``


    - **output_dir** *(PathOrStr)*:

        Output directory.

        default: ``None``


    - **source** *(str)*:

        Source to extract data from.

        default: ``nxcals``


    - **time** *(str)*:

        UTC Time as 'Y-m-d H:M:S.f' format.

        default: ``None``

    """
    if opt.output_dir is None:
        opt.log = True

    AccDT = AcceleratorDatetime[opt.accel]
    acc_time = AccDT.now() if opt.time is None else AccDT.from_utc_string(opt.time)

    beamprocess_info = _get_beamprocess(acc_time, opt.accel, opt.source)

    optics_info, knob_definitions, trims = None, None, None
    try:
        optics_info = _get_optics(acc_time, beamprocess_info.Name, beamprocess_info.StartTime)
    except ValueError as e:
        LOG.error(str(e))
    else:
        trims = LSA.find_trims_at_time(beamprocess_info.Object, opt.knobs, acc_time, opt.accel)
        knobs_definitions = _get_knob_definitions(opt.knob_definitions, opt.knobs, optics_info.Name)

    if opt.log:
        log_summary(acc_time, beamprocess_info, optics_info, trims)

    if opt.output_dir is not None:
        out = Path(opt.output_dir)
        out.mkdir(parents=True, exist_ok=True)
        write_summary(out, acc_time, beamprocess_info, optics_info, trims)
        if knob_definitions is not None:
            write_knob_defitions(out, knobs_definitions)

    return acc_time, beamprocess_info, optics_info, trims, knobs_definitions


# Output #######################################################################


def log_summary(acc_time: AccDatetime, bp_info: DotDict,
                optics_info: DotDict = None, trims: Dict[str, float] = None):
    """Log the summary.

    Args:
        acc_time (AccDatetime): User given Time
        bp_info (DotDict): BeamProcess Info Dictionary
        optics_info (DotDict): Optics Info Dictionary
        trims (dict): Trims key-value dictionary
    """
    summary = (
        "\n----------- Summary ---------------------\n"
        f"Given Time:   {acc_time.utc_string[:-7]} UTC\n"
        f"Fill:         {bp_info.Fill:d}\n"
        f"Beamprocess:  {bp_info.Name}\n"
        f"  Start:      {bp_info.StartTime.utc_string[:-7]} UTC\n"
        f"  Context:    {bp_info.ContextCategory}\n"
        f"  Descr.:     {bp_info.Description}\n"
    )
    if optics_info is not None:
        summary += (
            f"Optics:       {optics_info.Name}\n"
            f"  Start:      {optics_info.StartTime.utc_string[:-7]} UTC\n"
        )

    if trims is not None:
        summary += "----------- Trims -----------------------\n"
        for trim, value in trims.items():
            summary += f"{trim:30s}: {value:g}\n"

    summary += "-----------------------------------------\n\n"
    LOG.info(summary)


def write_summary(
    output_path: Path, acc_time: AccDatetime, bp_info: DotDict,
    optics_info: DotDict = None, trims: Dict[str, float] = None
):
    """Write summary into a ``tfs`` file.

    Args:
        output_path (Path): Folder to write output file into
        acc_time (AccDatetime): User given Time
        bp_info (DotDict): BeamProcess Info Dictionary
        optics_info (DotDict): Optics Info Dictionary
        trims (dict): Trims key-value dictionary
    """
    if trims is not None:
        trims = trims.items()

    info_tfs = tfs.TfsDataFrame(trims, columns=[const.column_knob, const.column_value])
    info_tfs.headers = OrderedDict([
        ("Hint:",                           "All times given in UTC."),
        (const.head_time,                   acc_time.cern_utc_string()),
        (const.head_beamprocess,            bp_info.Name),
        (const.head_fill,                   bp_info.Fill),
        (const.head_beamprocess_start,      bp_info.StartTime.cern_utc_string()),
        (const.head_context_category,       bp_info.ContextCategory),
        (const.head_beamprcess_description, bp_info.Description),
        ])
    if optics_info is not None:
        info_tfs.headers.update(OrderedDict([
            (const.head_optics, optics_info.Name),
            (const.head_optics_start, optics_info.StartTime.cern_utc_string()),
        ]))
    tfs.write(output_path / const.info_name, info_tfs)


def write_knob_defitions(output_path: Path, definitions: dict):
    """Write Knob definitions into a **tfs** file."""
    for knob, definition in definitions.items():
        path = output_path / f"{knob.replace('/', '_')}{const.knobdef_suffix}"
        tfs.write(path, definition, save_index=LSA_COLUMN_NAME)


# Beamprocess ##################################################################


def _get_beamprocess(acc_time: AccDatetime, accel: str, source: str) -> DotDict:
    """Get the info about the active beamprocess at ``acc_time``."""
    fill_no, fill_bps = LSA.find_last_fill(acc_time, accel, source)
    beamprocess = LSA.find_active_beamprocess_at_time(acc_time)
    try:
        start_time = _get_beamprocess_start(fill_bps, acc_time, str(beamprocess))
    except ValueError as e:
        raise ValueError(f"In fill {fill_no} the {str(e)}") from e
    bp_info = LSA.get_beamprocess_info(beamprocess)
    bp_info.update({"Fill": fill_no, "StartTime": start_time})
    return DotDict(bp_info)


def _get_beamprocess_start(beamprocesses: Iterable[Tuple[float, str]], acc_time: AccDatetime, bp_name: str) -> AccDatetime:
    """
    Get the last beamprocess in the list of beamprocesses before dt_utc.
    Returns the start time of the beam-process in utc.
    """
    LOG.debug(f"Looking for beamprocess '{bp_name}' in fill before '{acc_time.cern_utc_string()}'")
    ts = acc_time.timestamp()
    for time, name in sorted(beamprocesses, key=lambda x: x[0], reverse=True):
        if time <= ts and name == bp_name:
            return acc_time.__class__.from_timestamp(time)
    raise ValueError(
        f"Beamprocess '{bp_name}' was not found."
    )


# Optics #######################################################################


def _get_optics(acc_time: AccDatetime, beamprocess: str, bp_start: AccDatetime) -> DotDict:
    """Get the info about the active optics at ``acc_time``."""
    optics_table = LSA.getOpticTable(beamprocess)
    optics, start_time = _get_last_optics(optics_table, beamprocess, bp_start, acc_time)
    return DotDict({"Name": optics, "StartTime": start_time})


def _get_last_optics(
    optics_table, bp: str, bp_start: AccDatetime, acc_time: AccDatetime
) -> (str, AccDatetime):
    """Get the name of the optics at the right time for current beam process."""
    ts = acc_time.timestamp() - bp_start.timestamp()
    item = None
    for item in reversed(optics_table):
        if item.time <= ts:
            break
    if item is None:
        raise ValueError(f"No optics found for beamprocess {bp}")
    return item.name, acc_time.__class__.from_timestamp(item.time + bp_start.timestamp())


# Knobs ########################################################################


def _get_knob_definitions(active: bool, knobs: list, optics: str):
    """Get knob definitions if switch is activated."""
    defs = {}
    if active:
        for knob in knobs:
            try:
                defs[knob] = LSA.get_knob_circuits(knob, optics)
            except IOError as e:
                LOG.warning(e.args[0])
    return defs


# Script Mode ##################################################################


if __name__ == "__main__":
    get_info()
