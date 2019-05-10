"""
Print Machine Settings Overview
--------------------------------

Prints an overview over the machine settings at a given time,
or the current settings, if no time is given.

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


:module: print_machine_settings_overview
:author: jdilly

"""
import argparse
import logging
import re
import tfs
from typing import Iterable

from data_extract.lsa import LSA
from utils.dict_tools import DotDict
from utils.logging_tools import setup_logger
from utils.time_tools import AcceleratorDatetime, AccDatetime
from constants import machine_settings_info as const
from collections import OrderedDict

LOG = logging.getLogger(__name__)


DEFAULT_BP_RE = '^(RAMP|SQUEEZE)[_-]'
"""str: Default regexp for interesting Beamprocesses. """


# Main #########################################################################


def get_info(time: str = None, knobs: Iterable[str] = (), bp_regexp: str = DEFAULT_BP_RE, accel: str = 'lhc',
             log: bool = True, out: str = None) -> (AccDatetime, DotDict, DotDict, dict):
    """
    Get info about Beamprocess, Optics and Knob-values at given time.

    Args:
        time (str): UTC time as Y-m-d H:M:S.f format. Default: *now*
        knobs (list): List of knob-names to check. Default: *all available*
        bp_regexp (str): Regular expression to filter beamprocesses. Default: ``'^(RAMP|SQUEEZE)[_-]'``
        accel (str): Accelerator name, as used in pjlsa. Default ``'lhc'``
        log (bool): Activates or Deactivates logging the results. Default: ``True``
        out (str): Path to the output folder. If not given, no output files. Default: ``None``

    """
    AccDT = AcceleratorDatetime[accel]
    acc_time = AccDT.now() if time is None else AccDT.from_utc_string(time)

    beamprocess_info = _get_beamprocess(acc_time, bp_regexp, accel)
    optics_info = _get_optics(acc_time, beamprocess_info.name, beamprocess_info.start)
    trims = LSA.find_trims_at_time(beamprocess_info.name, knobs, acc_time, accel)

    if log:
        log_summary(acc_time, beamprocess_info, optics_info, trims)

    if out is not None:
        write_summary(out, acc_time, beamprocess_info, optics_info, trims)

    return acc_time, beamprocess_info, optics_info, trims


# Output #######################################################################


def log_summary(acc_time: AccDatetime, bp_info: DotDict, o_info: DotDict, trims: dict):
    """ Log the summary. """
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


def write_summary(output_path: str, acc_time: AccDatetime, bp_info: DotDict, o_info: DotDict, trims: dict):
    """ Write summary into file. """
    info_tfs = tfs.TfsDataFrame(trims.items(), columns=[const.column_knob, const.column_value])
    info_tfs.headers = OrderedDict([
        (const.head_time,                   acc_time.utc_string()),
        (const.head_beamprocess,            bp_info.name),
        (const.head_fill,                   bp_info.fill),
        (const.head_beamprocess_start,      bp_info.start.utc_string()),
        (const.head_context_category,       bp_info.contextCategory),
        (const.head_beamprcess_description, bp_info.description),
        (const.head_optics,                 o_info.name),
        (const.head_optics_start,           o_info.start.utc_string()),
        ("Hint:",                           "All times given in UTC.")
    ])
    tfs.write(output_path, info_tfs)


# Beamprocess ##################################################################


def _get_beamprocess(acc_time: AccDatetime, regexp: str, accel: str) -> DotDict:
    """ Get the info about the active beamprocess at acc_time. """
    fill_no, fill_bps = LSA.find_last_fill(acc_time, accel)
    beamprocess, start_time = _get_last_beamprocess(fill_bps, acc_time, regexp)
    bp_info = LSA.get_beamprocess_info(beamprocess)
    bp_info.update({
        'name': beamprocess,
        'fill': fill_no,
        'start': start_time,
    })
    return DotDict(bp_info)


def _get_last_beamprocess(bps, acc_time: AccDatetime, regexp: str) -> (str, AccDatetime):
    """ Get the last beamprocess in the list of beamprocesses before dt_utc.
    Also returns the start time of the beam-process in utc.
    """
    ts = acc_time.timestamp()
    LOG.debug(f"Looking for beamprocesses before '{acc_time.utc_string()}', matching '{regexp}'")
    time, name = None, None
    reg = re.compile(regexp, re.IGNORECASE)
    for time, name in reversed(bps):
        if time <= ts and reg.search(name) is not None:
            break
    if time is None:
        raise ValueError(f"No relevant beamprocess found in the fill before {acc_time.utc_string()}")
    return name, acc_time.__class__.from_timestamp(time)


# Optics #######################################################################


def _get_optics(acc_time: AccDatetime, beamprocess: str, bp_start: AccDatetime) -> DotDict:
    """ Get the info about the active optics at acc_time. """
    optics_table = LSA.getOpticTable(beamprocess)
    optics, start_time = _get_last_optics(optics_table, beamprocess, bp_start, acc_time)
    return DotDict({
        'name': optics,
        'start': start_time,
    })


def _get_last_optics(optics_table, bp: str, bp_start: AccDatetime, acc_time: AccDatetime ) -> (str, AccDatetime):
    """ Get the name of the optics at the right time for current beam process. """
    ts = acc_time.timestamp() - bp_start.timestamp()
    item = None
    for item in reversed(optics_table):
        if item.time <= ts:
            break
    if item is None:
        raise ValueError(f"No optics found for beamprocess {bp}")
    return item.name, acc_time.__class__.from_timestamp(item.time + bp_start.timestamp())


# Script Mode ##################################################################


def get_options() -> dict:
    """ Parse Commandline Arguments and return them as options. """
    args = argparse.ArgumentParser()
    args.add_argument("--time", "-t",      default=None,          type=str,  help="Time as 'Y-m-d H:M:S.f' format.")
    args.add_argument("--knobs", "-k",     default=(), nargs="+", type=str,  help="List of knobnames.")
    args.add_argument("--bp_regexp", "-r", default=DEFAULT_BP_RE, type=str,  help="Beamprocess regexp filter.")
    args.add_argument("--accel", "-a",     default='lhc',         type=str,  help="Accelerator name.")
    args.add_argument("--out", "-o",       default=None,          type=str,  help="Output path.")
    return args.parse_args().__dict__


if __name__ == '__main__':
    setup_logger()
    get_info(**get_options())
