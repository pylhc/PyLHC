"""
Print Machine Settings Overview
-------------------------------

Prints an overview over the machine settings at a provided given time, or the current settings if
no time is given.
If an output path is given, all info will be written into tfs files,
otherwise a summary is logged into console.

Knob values can be extracted and the knob definition gathered.
For brevity reasons, this data is not logged into the summary in the console.
If a start time is given, the trim history for the given knobs can be written out as well.
This data is also not logged.

Can be run from command line, parameters as given in :meth:`pylhc.machine_settings_info.get_info`.
All gathered data is returned, if this function is called from python.

.. code-block:: none

   usage: machine_settings_info.py [-h] [--time TIME] [--start_time START_TIME]
                                   [--knobs KNOBS [KNOBS ...]] [--accel ACCEL]
                                   [--beamprocess BEAMPROCESS] [--output_dir OUTPUT_DIR]
                                   [--knob_definitions] [--source SOURCE] [--log]

  optional arguments:
  -h, --help            show this help message and exit
  --time TIME           UTC Time as 'Y-m-d H:M:S.f' or ISO format or AccDatetime object.
                        Acts as point in time or end time (if ``start_time`` is given).
  --start_time START_TIME
                        UTC Time as 'Y-m-d H:M:S.f' or ISO format or AccDatetime object.
                        Defines the beginning of the time-range.
  --knobs KNOBS [KNOBS ...]
                        List of knobnames. If `None` (or omitted) no knobs will be extracted.
                        If it is just the string ``'all'``, all knobs will be extracted
                        (can be slow). Use the string ``'default'`` for pre-defined knobs
                        of interest.
  --accel ACCEL         Accelerator name.
  --beamprocess BEAMPROCESS
                        Manual override for the Beamprocess
                        (otherwise taken at the given ``time``)
  --output_dir OUTPUT_DIR
                        Output directory.
  --knob_definitions    Set to extract knob definitions.
  --source SOURCE       Source to extract data from.
  --log                 Write summary into log (automatically done if no output path is given).


:author: jdilly
"""

from collections import OrderedDict, namedtuple
from collections.abc import Iterable
from pathlib import Path

import tfs
from generic_parser import DotDict, EntryPointParameters, entrypoint
from generic_parser.entry_datatypes import get_instance_faker_meta
from omc3.knob_extractor import KNOB_CATEGORIES, name2lsa
from omc3.utils import logging_tools
from omc3.utils.iotools import PathOrStr
from omc3.utils.time_tools import AccDatetime, AcceleratorDatetime

from pylhc.constants import machine_settings_info as const
from pylhc.data_extract.lsa import COL_NAME as LSA_COLUMN_NAME
from pylhc.data_extract.lsa import LSA

LOG = logging_tools.get_logger(__name__)


class AccDatetimeOrStr(metaclass=get_instance_faker_meta(AccDatetime, str)):
    """A class that accepts AccDateTime and strings."""

    def __new__(cls, value):
        if isinstance(value, str):
            value = value.strip(
                "'\""
            )  # behavior like dict-parser, IMPORTANT FOR EVERY STRING-FAKER
        return value


# Main #########################################################################


def _get_params() -> dict:
    """Parse Commandline Arguments and return them as options."""
    return EntryPointParameters(
        time={
            "default": None,
            "type": AccDatetimeOrStr,
            "help": (
                "UTC Time as 'Y-m-d H:M:S.f' or ISO format or AccDatetime object."
                " Acts as point in time or end time (if ``start_time`` is given)."
            ),
        },
        start_time={
            "default": None,
            "type": AccDatetimeOrStr,
            "help": (
                "UTC Time as 'Y-m-d H:M:S.f' or ISO format or AccDatetime object."
                " Defines the beginning of the time-range."
            ),
        },
        knobs={
            "default": None,
            "nargs": "+",
            "type": str,
            "help": "List of knobnames. "
            "If `None` (or omitted) no knobs will be extracted. "
            "If it is just the string ``'all'``, "
            "all knobs will be extracted (can be slow). "
            "Use the string ``'default'`` for pre-defined knobs of interest.",
        },
        accel={"default": "lhc", "type": str, "help": "Accelerator name."},
        beamprocess={
            "type": str,
            "help": ("Manual override for the Beamprocess (otherwise taken at the given ``time``)"),
        },
        output_dir={"default": None, "type": PathOrStr, "help": "Output directory."},
        knob_definitions={"action": "store_true", "help": "Set to extract knob definitions."},
        source={"type": str, "default": "nxcals", "help": "Source to extract data from."},
        log={
            "action": "store_true",
            "help": "Write summary into log (automatically done if no output path is given).",
        },
    )


@entrypoint(_get_params(), strict=True)
def get_info(opt) -> dict[str, object]:
    """
     Get info about **Beamprocess**, **Optics** and **Knobs** at given time.

     Keyword Args:

    *--Optional--*

     - **accel** *(str)*:

         Accelerator name.

         default: ``lhc``


     - **beamprocess** *(str)*:

         Manual override for the Beamprocess
         (otherwise taken at the given ``time``)

         default: ``None``


     - **knob_definitions**:

         Set to extract knob definitions.

         action: ``store_true``


     - **knobs** *(str)*:

         List of knobnames.
         If `None` (or omitted) no knobs will be extracted.
         If it is just the string ``'all'``,
         all knobs will be extracted (can be slow).
         Use the string ``'default'`` for pre-defined knobs of interest.
         If this is called from python, the strings need
         to be put as single items into a list.

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


     - **start_time** *(AccDatetime, str)*:

         UTC Time as 'Y-m-d H:M:S.f' format or AccDatetime object.
         Defines the beginning of the time-range.

         default: ``None``


     - **time** *(AccDatetime, str)*:

         UTC Time as 'Y-m-d H:M:S.f' format or AccDatetime object.
         Acts as point in time or end time (if ``start_time`` is given).

         default: ``None``

     Returns:
         dict: Dictionary containing the given ``time`` and ``start_time``,
         the extracted ``beamprocess``-info and ``optics``-info, the
         ``trim_histories`` and current (i.e. at given ``time``) ``trims``
         and the ``knob_definitions``, if extracted.

    """
    if opt.output_dir is None:
        opt.log = True

    acc_time, acc_start_time = _get_times(opt.time, opt.start_time, opt.accel)
    beamprocess_info = _get_beamprocess(acc_time, opt.accel, opt.source, opt.beamprocess)

    optics_info, knob_definitions, trim_histories, trims = None, None, None, None
    try:
        optics_info = _get_optics(acc_time, beamprocess_info.Name, beamprocess_info.StartTime)
    except ValueError as e:
        LOG.error(str(e))
    else:
        if opt.knobs is not None:
            if len(opt.knobs) == 1 and opt.knobs[0].lower() == "all":
                opt.knobs = []  # will extract all knobs in get_trim_history
            if len(opt.knobs) == 1 and opt.knobs[0].lower() == "default":
                opt.knobs = [
                    name2lsa(knob) for category in KNOB_CATEGORIES.values() for knob in category
                ]

            trim_histories = LSA.get_trim_history(
                beamprocess_info.Object,
                opt.knobs,
                start_time=acc_start_time,
                end_time=acc_time,
                accelerator=opt.accel,
            )
            trims = _get_last_trim(trim_histories)

        if opt.knob_definitions:
            if trim_histories:  # this now works with 'all'/`default`. Might create a lot of files
                knob_definitions = _get_knob_definitions(trim_histories.keys(), optics_info.Name)
            else:
                LOG.error("Writing out knob definitions requested, but no knobs extracted.")

    if opt.log:
        log_summary(acc_time, beamprocess_info, optics_info, trims)

    if opt.output_dir is not None:
        out_path = Path(opt.output_dir)
        out_path.mkdir(parents=True, exist_ok=True)
        write_summary(out_path, opt.accel, acc_time, beamprocess_info, optics_info, trims)

        if trim_histories and acc_start_time:
            write_trim_histories(
                out_path,
                trim_histories,
                opt.accel,
                acc_time,
                acc_start_time,
                beamprocess_info,
                optics_info,
            )

        if knob_definitions:
            write_knob_defitions(out_path, knob_definitions)

    return {
        "time": acc_time,
        "start_time": acc_start_time,
        "beamprocess": beamprocess_info,
        "optics": optics_info,
        "trim_histories": trim_histories,
        "trims": trims,
        "knob_definitions": knob_definitions,
    }


# Output #######################################################################


def log_summary(
    acc_time: AccDatetime,
    bp_info: DotDict,
    optics_info: DotDict = None,
    trims: dict[str, float] = None,
):
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
    output_path: Path,
    accel: str,
    acc_time: AccDatetime,
    bp_info: DotDict,
    optics_info: DotDict = None,
    trims: dict[str, float] = None,
):
    """Write summary into a ``tfs`` file.

    Args:
        output_path (Path): Folder to write output file into
        accel (str): Name of the accelerator
        acc_time (AccDatetime): User given Time
        bp_info (DotDict): BeamProcess Info Dictionary
        optics_info (DotDict): Optics Info Dictionary
        trims (dict): Trims key-value dictionary
    """
    if trims is not None:
        trims = trims.items()

    info_tfs = tfs.TfsDataFrame(trims, columns=[const.column_knob, const.column_value])
    info_tfs.headers = OrderedDict(
        [
            ("Hint:", "All times given in UTC."),
            (const.head_accel, accel),
            (const.head_time, acc_time.cern_utc_string()),
            (const.head_beamprocess, bp_info.Name),
            (const.head_fill, bp_info.Fill),
            (const.head_beamprocess_start, bp_info.StartTime.cern_utc_string()),
            (const.head_context_category, bp_info.ContextCategory),
            (const.head_beamprcess_description, bp_info.Description),
        ]
    )
    if optics_info is not None:
        info_tfs.headers.update(
            OrderedDict(
                [
                    (const.head_optics, optics_info.Name),
                    (const.head_optics_start, optics_info.StartTime.cern_utc_string()),
                ]
            )
        )
    tfs.write(output_path / const.info_name, info_tfs)


def write_knob_defitions(output_path: Path, definitions: dict):
    """Write Knob definitions into a **tfs** file."""
    for knob, definition in definitions.items():
        path = output_path / f"{knob.replace('/', '_')}{const.knobdef_suffix}"
        tfs.write(path, definition, save_index=LSA_COLUMN_NAME)


def write_trim_histories(
    output_path: Path,
    trim_histories: dict[str, namedtuple],
    accel: str,
    acc_time: AccDatetime = None,
    acc_start_time: AccDatetime = None,
    bp_info: DotDict = None,
    optics_info: DotDict = None,
):
    """Write the trim histories into tfs files.
    There are two time columns, one with timestamps as they are usually easier to handle
    and one with the UTC-string, as they are more human-readable.

    Args:
        output_path (Path): Folder to write output file into
        trim_histories (dict): trims histories as extracted via LSA.get_trim_history()
        accel (str): Name of the accelerator
        acc_time (AccDatetime): User given (End)Time
        acc_start_time (AccDatetime): User given Start Time
        bp_info (DotDict): BeamProcess Info Dictionary
        optics_info (DotDict): Optics Info Dictionary
    """
    AccDT = AcceleratorDatetime[accel]  # noqa: N806

    # Create headers with basic info ---
    headers = OrderedDict([("Hint:", "All times are given in UTC."), (const.head_accel, accel)])

    if acc_start_time:
        headers.update({const.head_start_time: acc_start_time.cern_utc_string()})

    if acc_time:
        headers.update({const.head_end_time: acc_time.cern_utc_string()})

    if bp_info:
        headers.update(
            {
                const.head_beamprocess: bp_info.Name,
                const.head_fill: bp_info.Fill,
            }
        )

    if optics_info:
        headers.update({const.head_optics: optics_info.Name})

    # Write trim history per knob ----
    for knob, trim_history in trim_histories.items():
        trims_tfs = tfs.TfsDataFrame(
            headers=headers, columns=[const.column_time, const.column_timestamp, const.column_value]
        )
        for timestamp, value in zip(trim_history.time, trim_history.data):
            time = AccDT.from_timestamp(timestamp).cern_utc_string()
            try:
                len(value)
            except TypeError:
                # single value (as it should be)
                trims_tfs.loc[len(trims_tfs), :] = (time, timestamp, value)
            else:
                # multiple values (probably weird)
                LOG.debug("Multiple values in trim for {knob} at {time}.")
                for item in value:
                    trims_tfs.loc[len(trims_tfs), :] = (time, timestamp, item)

        path = output_path / f"{knob.replace('/', '_')}{const.trimhistory_suffix}"
        tfs.write(path, trims_tfs)


# Beamprocess ##################################################################


def _get_beamprocess(
    acc_time: AccDatetime, accel: str, source: str, beamprocess: str = None
) -> DotDict:
    """Get the info about the active beamprocess at ``acc_time`` or the given one."""
    if beamprocess is None:
        beamprocess = LSA.find_active_beamprocess_at_time(acc_time, accelerator=accel)
    bp_info = LSA.get_beamprocess_info(beamprocess)
    bp_info = DotDict(bp_info)

    fill_no, fill_bps = LSA.find_last_fill(acc_time, accelerator=accel, source=source)
    bp_info.Fill = fill_no

    try:
        start_time = _get_beamprocess_start(fill_bps, acc_time, bp_info.Name)
    except ValueError as e:
        raise ValueError(f"In fill {fill_no} the {str(e)}") from e
    bp_info.StartTime = start_time

    LOG.debug(
        f"Beamprocess {bp_info.Name} in fill {bp_info.Fill} started at time {bp_info.StartTime}."
    )
    return DotDict(bp_info)


def _get_beamprocess_start(
    beamprocesses: Iterable[tuple[float, str]], acc_time: AccDatetime, bp_name: str
) -> AccDatetime:
    """
    Get the last beamprocess in the list of beamprocesses before dt_utc.
    Returns the start time of the beam-process in utc.
    """
    LOG.debug(f"Looking for beamprocess '{bp_name}' in fill before '{acc_time.cern_utc_string()}'")
    ts = acc_time.timestamp()
    for time, name in sorted(beamprocesses, key=lambda x: x[0], reverse=True):
        if time <= ts and name == bp_name:
            LOG.debug(f"Found start for beamprocess '{bp_name}' at timestamp {time}.")
            return acc_time.__class__.from_timestamp(time)
    raise ValueError(f"Beamprocess '{bp_name}' was not found.")


# Optics #######################################################################


def _get_optics(acc_time: AccDatetime, beamprocess: str, bp_start: AccDatetime) -> DotDict:
    """Get the info about the active optics at ``acc_time``."""
    LOG.debug(f"Getting optics for {beamprocess} at time {acc_time.utc_string}")
    optics_table = LSA.getOpticTable(beamprocess)
    optics, start_time = _get_last_optics(optics_table, beamprocess, bp_start, acc_time)
    LOG.debug(f"Optics {optics} extracted at time {start_time.utc_string}.")
    return DotDict({"Name": optics, "StartTime": start_time})


def _get_last_optics(
    optics_table, bp: str, bp_start: AccDatetime, acc_time: AccDatetime
) -> tuple[str, AccDatetime]:
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


def _get_knob_definitions(knobs: list, optics: str):
    """Get knob definitions."""
    LOG.debug(f"Extracting knob definitions for {len(knobs)} in optics '{optics}'.")
    defs = {}
    for knob in knobs:
        try:
            defs[knob] = LSA.get_knob_circuits(knob, optics)
        except OSError as e:
            LOG.warning(e.args[0])
    return defs


def _get_last_trim(trims: dict) -> dict:
    """Returns the last trim in the trim history.

    Args:
        trims (dict): trim history as extracted via LSA.get_trim_history()

    Returns:
        Dictionary of knob names and their values.
    """
    LOG.debug("Extracting last trim from found trim histories.")
    trim_dict = {trim: trims[trim].data[-1] for trim in trims}  # return last set value
    for trim, value in trim_dict.items():
        try:
            trim_dict[trim] = value.flatten()[-1]  # the very last entry ...
        except AttributeError:
            continue  # single value, as expected
        else:
            LOG.warning(f"Trim {trim} hat multiple data entries {value}, taking only the last one.")
    return trim_dict


# Other ########################################################################


def _get_times(time: str | AccDatetime, start_time: str | AccDatetime, accel: str):
    """Returns acc_time and acc_start_time parameters depending on the user input."""
    acc_dt = AcceleratorDatetime[accel]

    def get_time(t, default=None):
        if t is None:
            return default

        try:
            return acc_dt.from_utc_string(t)
        except ValueError:
            # time was in wrong format
            return acc_dt.fromisoformat(t)
        except TypeError:
            # is already AccDatetime object
            return t

    time = get_time(time, default=acc_dt.now())
    start_time = get_time(start_time, default=None)
    return time, start_time


# Script Mode ##################################################################


if __name__ == "__main__":
    get_info()
