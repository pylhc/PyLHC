"""
Print Machine Settings Overview
--------------------------------

Prints an overview over the machine settings at a given time,
or the current settings, if no time is given.

:author: jdilly

"""
import argparse
import logging
import re
from typing import Iterable

from data_extract.pylsa import LSAClient
from utils.logging_tools import setup_logger
from utils.time_tools import AcceleratorDatetime

LOG = logging.getLogger(__name__)


DEFAULT_BP_RE = '^(RAMP|SQUEEZE)[_-]'
"""str: Default regexp for interesting Beamprocesses. """


def get_options():
    """ Parse Commandline Arguments """
    args = argparse.ArgumentParser()
    args.add_argument("--time", "-t", default=None)
    args.add_argument("--knobs", "-k", default=[], nargs="+")
    return args.parse_args()


def main(time: str = None, knobs: Iterable[str] = (), bp_regexp: str = DEFAULT_BP_RE, accel: str = 'lhc'):
    """
    Write an overview of Beamprocess, Optics and Knob-values at given time.

    Args:
        time (str): UTC time as YY-mm-dd HH:MM:S.f format.
        knobs (list): List of knob-names to check (default: all available)
        bp_regexp (str): Regular expression to filter beamprocesses (default: '^(RAMP|SQUEEZE)[_-]')
        accel (str): Accelerator (default 'lhc')

    """
    AccDT = AcceleratorDatetime[accel]
    if time is None:
        acc_time = AccDT.now()
    else:
        acc_time = AccDT.from_utc_string(time)

    lsa = LSAClient()

    # get beamprocess
    fill_no, fill_bps = lsa.find_last_fill(acc_time, accel)
    beamprocess, bp_start = _get_last_beamprocess(fill_bps, acc_time, bp_regexp)
    bp_info = lsa.get_beamprocess_info(beamprocess)

    # get optics
    optics_table = lsa.getOpticTable(beamprocess)
    optics, optics_start = _get_last_optics(optics_table, beamprocess, bp_start, acc_time)

    # get trims
    trims = lsa.find_trims_at_time(beamprocess, knobs, acc_time, accel)

    # print summary
    summary = (
        "\n----------- Summary ---------------------\n"
        f"Given Time:   {acc_time.utc_string()}\n"
        f"Fill:         {fill_no:d}\n"
        f"Beamprocess:  {beamprocess}\n"
        f"  Start:      {bp_start.utc_string()}\n"
        f"  Context:    {bp_info['contextCategory']}\n"
        f"  Descr.:     {bp_info['description']}\n"
        f"Optics:       {optics}\n"
        f"  Start:      {optics_start.utc_string()}\n"
        "-----------------------------------------\n"
    )
    for trim, value in trims.items():
        summary += f"{trim:30s}: {value:g}\n"
    summary += "-----------------------------------------\n\n"
    LOG.info(summary)


def _get_last_beamprocess(bps, acc_time, regexp=''):
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


def _get_last_optics(optics_table, bp, bp_start, acc_time):
    """ Get the name of the optics at the right time for current beam process. """
    ts = acc_time.timestamp() - bp_start.timestamp()
    item = None
    for item in reversed(optics_table):
        if item.time <= ts:
            break
    if item is None:
        raise ValueError(f"No optics found for beamprocess {bp}")
    return item.name, acc_time.__class__.from_timestamp(item.time + bp_start.timestamp())


if __name__ == '__main__':
    setup_logger()
    main(**get_options().__dict__)
