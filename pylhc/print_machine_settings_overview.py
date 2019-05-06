"""
Print Machine Settings Overview
--------------------------------

Prints an overview over the machine settings at a given time,
or the current settings, if no time is given.

:author: jdilly

"""
import argparse
import logging
import ruamel_yaml as yaml
import logging.config
from utils.time_tools import AcceleratorDatetime

from data_extract.pylsa import LSAClient


LOG = logging.getLogger(__name__)


def get_options():
    """ Parse Commandline Arguments """
    args = argparse.ArgumentParser()
    args.add_argument("--time", "-t", default=None)
    args.add_argument("--knobs", "-k", default=[], nargs="+")
    return args.parse_args()


def main(time=None, knobs=(), accel='lhc'):
    """
    Write an overview of Beamprocess, Optics and Knob-values at given time.

    Args:
        time: UTC time as YY-mm-dd HH:MM:S.f format.
        knobs: List of knob-names to check
        accel: Accelerator

    """
    AccDT = AcceleratorDatetime[accel]
    if time is None:
        acc_time = AccDT.now()
    else:
        acc_time = AccDT.from_utc_string(time)

    lsa = LSAClient()

    fill_no, fill_beamprocesses = lsa.get_last_fill(acc_time, accel)
    beamprocess, bp_start = _get_last_beamprocess(fill_beamprocesses, acc_time)
    optics_table = lsa.getOpticTable(beamprocess)
    optics, optics_start = _get_last_optics(optics_table, bp_start, acc_time)
    trims = lsa.get_trims_at_time(beamprocess, knobs, acc_time, accel)

    LOG.info("\n--- Summary -------------------------------")
    LOG.info(f"Given Time:   {acc_time.utc_string()}")
    LOG.info(f"Fill:         {fill_no:d}")
    LOG.info(f"Beamprocess:  {beamprocess}")
    LOG.info(f"BP-Start:     {bp_start.utc_string()}")
    LOG.info(f"Optics:       {optics}")
    LOG.info(f"Optics-Start: {optics_start.utc_string()}")
    for trim, value in trims.items():
        LOG.info(f"{trim:30s}: {value:g}")


def _get_last_beamprocess(beamprocesses, acc_time):
    """ Get the last beamprocess in the list of beamprocesses before dt_utc.
    Also returns the start time of the beam-process in utc.
    """
    ts = acc_time.timestamp()
    for time, name in reversed(beamprocesses):
        if time <= ts:
            break
    return name, acc_time.__class__.from_timestamp(time)


def _get_last_optics(optics_table, bp_start, acc_time):
    """ Get the name of the optics at the right time for current beam process. """
    ts = acc_time.timestamp() - bp_start.timestamp()
    for item in reversed(optics_table):
        if item.time <= ts:
            break
    return item.name, acc_time.__class__.from_timestamp(item.time + bp_start.timestamp())


def _setup_logger():
    """ Setup logging.

    Might be replaced in the future with logging_tools.
    """
    with open('logging_config.yml') as stream:
        logging.config.dictConfig(yaml.safe_load(stream))


if __name__ == '__main__':
    _setup_logger()
    main(**get_options().__dict__)
