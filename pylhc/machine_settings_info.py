"""
Print Machine Settings Overview
-------------------------------

Prints an overview over the machine settings at a provided given time, or the current settings if
no time is given.

Can be run from command line, parameters as given in :meth:`print_machine_settings_overview.main`.

.. code-block:: none

  print_machine_settings_overview.py [-h] [--time TIME]
                                          [--knobs KNOBS [KNOBS ...]]
                                          [--bp_regexp BP_REGEXP]
                                          [--accel ACCEL]
                                          [--out OUT]

  optional arguments:
    -h, --help            show this help message and exit
    --time TIME, -t TIME  Time as 'Y-m-d H:M:S.f' format.
    --knobs KNOBS [KNOBS ...], -k KNOBS [KNOBS ...]
                          List of knobnames.
    --bp_regexp BP_REGEXP, -r BP_REGEXP
                          Beamprocess regexp filter.
    --accel ACCEL, -a ACCEL
                          Accelerator name.
    --out OUT, -o OUT     Output path.
"""
import re
from collections import OrderedDict
from pathlib import Path

import tfs
from generic_parser import DotDict, EntryPointParameters, entrypoint
from omc3.utils import logging_tools
from omc3.utils.time_tools import AccDatetime, AcceleratorDatetime

from pylhc.constants import machine_settings_info as const
from pylhc.data_extract.lsa import COL_NAME as lsa_col_name, LSA

LOG = logging_tools.get_logger(__name__)


DEFAULT_BP_RE = "^(RAMP|SQUEEZE)[_-]"
"""str: Default regexp for interesting Beamprocesses. """


# Main #########################################################################


def _get_params() -> dict:
    """Parse Commandline Arguments and return them as options."""
    return EntryPointParameters(
        time=dict(
            flags=["--time", "-t"], default=None, type=str, help="Time as 'Y-m-d H:M:S.f' format."
        ),
        knobs=dict(
            flags=["--knobs", "-k"], default=(), nargs="+", type=str, help="List of knobnames."
        ),
        bp_regexp=dict(
            flags=["--bp_regexp", "-r"],
            default=DEFAULT_BP_RE,
            type=str,
            help="Beamprocess regexp filter.",
        ),
        accel=dict(flags=["--accel", "-a"], default="lhc", type=str, help="Accelerator name."),
        out=dict(flags=["--out", "-o"], default=None, type=str, help="Output path."),
        knob_def=dict(
            flags=["--knob_def", "-d"], action="store_true", help="Set to extract knob definitions."
        ),
        log=dict(
            flags=["--log", "-l"],
            action="store_true",
            help="Write summary into log (automatically done if no output path is given).",
        ),
    )


@entrypoint(_get_params(), strict=True)
def get_info(opt) -> (AccDatetime, DotDict, DotDict, dict, dict):
    """
    Get info about **Beamprocess**, **Optics** and **Knobs** at given time.

    Keyword Args:
        *--Optional--*
        - **accel** *(str)*: Accelerator name.

          Flags: **['--accel', '-a']**
          Default: ``lhc``
        - **bp_regexp** *(str)*: Beamprocess regexp filter.

          Flags: **['--bp_regexp', '-r']**
          Default: ``^(RAMP|SQUEEZE)[_-]``
        - **knob_def**: Set to extract knob definitions.

          Flags: **['--knob_def', '-d']**
          Action: ``store_true``
        - **knobs** *(str)*: List of knobnames.

          Flags: **['--knobs', '-k']**
          Default: ``()``
        - **log**: Write summary into log (automatically done if no output path is given).

          Flags: **['--log', '-l']**
          Action: ``store_true``
        - **out** *(str)*: Output path.

          Flags: **['--out', '-o']**
          Default: ``None``
        - **time** *(str)*: Time as 'Y-m-d H:M:S.f' format.

          Flags: **['--time', '-t']**
          Default: ``None``
    """
    if opt.out is None:
        opt.log = True

    AccDT = AcceleratorDatetime[opt.accel]
    acc_time = AccDT.now() if opt.time is None else AccDT.from_utc_string(opt.time)

    beamprocess_info = _get_beamprocess(acc_time, opt.bp_regexp, opt.accel)
    optics_info = _get_optics(acc_time, beamprocess_info.name, beamprocess_info.start)
    trims = LSA.find_trims_at_time(beamprocess_info.name, opt.knobs, acc_time, opt.accel)
    knobs_definitions = _get_knob_definitions(opt.knob_def, opt.knobs, optics_info.name)

    if opt.log:
        log_summary(acc_time, beamprocess_info, optics_info, trims)

    if opt.out is not None:
        write_summary(opt.out, acc_time, beamprocess_info, optics_info, trims)
        write_knob_defitions(opt.out, knobs_definitions)

    return acc_time, beamprocess_info, optics_info, trims, knobs_definitions


# Output #######################################################################


def log_summary(acc_time: AccDatetime, bp_info: DotDict, o_info: DotDict, trims: dict):
    """Log the summary."""
    summary = (
        "\n----------- Summary ---------------------\n"
        f"Given Time:   {acc_time.utc_string()}\n"
        f"Fill:         {bp_info.fill:d}\n"
        f"Beamprocess:  {bp_info.name}\n"
        f"  Start:      {bp_info.start.utc_string()}\n"
        f"  Context:    {bp_info.contextCategory}\n"
        f"  Descr.:     {bp_info.description}\n"
        f"Optics:       {o_info.name}\n"
        f"  Start:      {o_info.start.utc_string()}\n"
        "-----------------------------------------\n"
    )
    for trim, value in trims.items():
        summary += f"{trim:30s}: {value:g}\n"
    summary += "-----------------------------------------\n\n"
    LOG.info(summary)


def write_summary(
    output_path: str, acc_time: AccDatetime, bp_info: DotDict, o_info: DotDict, trims: dict
):
    """Write summary into a **tfs** file."""
    info_tfs = tfs.TfsDataFrame(trims.items(), columns=[const.column_knob, const.column_value])
    info_tfs.headers = OrderedDict(
        [
            (const.head_time, acc_time.cern_utc_string()),
            (const.head_beamprocess, bp_info.name),
            (const.head_fill, bp_info.fill),
            (const.head_beamprocess_start, bp_info.start.cern_utc_string()),
            (const.head_context_category, bp_info.contextCategory),
            (const.head_beamprcess_description, bp_info.description),
            (const.head_optics, o_info.name),
            (const.head_optics_start, o_info.start.cern_utc_string()),
            ("Hint:", "All times given in UTC."),
        ]
    )
    tfs.write(str(Path(output_path, const.info_name)), info_tfs)


def write_knob_defitions(output_path: str, definitions: dict):
    """Write Knob definitions into a **tfs** file."""
    for knob, definition in definitions.items():
        path = Path(output_path, f"{knob.replace('/', '_')}{const.knobdef_suffix}")
        tfs.write(str(path), definition, save_index=lsa_col_name)


# Beamprocess ##################################################################


def _get_beamprocess(acc_time: AccDatetime, regexp: str, accel: str) -> DotDict:
    """Get the info about the active beamprocess at ``acc_time``."""
    fill_no, fill_bps = LSA.find_last_fill(acc_time, accel)
    beamprocess, start_time = _get_last_beamprocess(fill_bps, acc_time, regexp)
    bp_info = LSA.get_beamprocess_info(beamprocess)
    bp_info.update({"name": beamprocess, "fill": fill_no, "start": start_time})
    return DotDict(bp_info)


def _get_last_beamprocess(bps, acc_time: AccDatetime, regexp: str) -> (str, AccDatetime):
    """
    Get the last beamprocess in the list of beamprocesses before dt_utc.
    Also returns the start time of the beam-process in utc.
    """
    ts = acc_time.timestamp()
    LOG.debug(
        f"Looking for beamprocesses before '{acc_time.cern_utc_string()}', matching '{regexp}'"
    )
    time, name = None, None
    reg = re.compile(regexp, re.IGNORECASE)
    for time, name in reversed(bps):
        if time <= ts and reg.search(name) is not None:
            break
    if time is None:
        raise ValueError(
            f"No relevant beamprocess found in the fill before {acc_time.cern_utc_string()}"
        )
    return name, acc_time.__class__.from_timestamp(time)


# Optics #######################################################################


def _get_optics(acc_time: AccDatetime, beamprocess: str, bp_start: AccDatetime) -> DotDict:
    """Get the info about the active optics at ``acc_time``."""
    optics_table = LSA.getOpticTable(beamprocess)
    optics, start_time = _get_last_optics(optics_table, beamprocess, bp_start, acc_time)
    return DotDict({"name": optics, "start": start_time})


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
