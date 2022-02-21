"""
Timber
------

This module provides useful functions to conveniently wrap the functionality of ``pytimber``.
"""
import logging

from omc3.utils.mock import cern_network_import
from omc3.utils.time_tools import AccDatetime

pytimber = cern_network_import("pytimber")

LOG = logging.getLogger(__name__)


def find_exact_time_for_beamprocess(acc_time: AccDatetime) -> AccDatetime:
    """
    Finds the last entry where HX:SRMP-POW equals 123. Which is, according to the online model
    KnobExtractor, the timing event for SQUEEZE or RAMP.
    I don't think this is correct (jdilly).
    """
    db = pytimber.LoggingDB(source="nxcals")
    t1, t2 = acc_time.sub(days=1).local_string(), acc_time.local_string()

    event_ts, event_val = db.get("HX:SRMP-POW", t1, t2)["HX:SRMP-POW"]
    event_ts = event_ts[event_val == 123.0]

    if len(event_ts) == 0:
        raise ValueError(
            f"No valid beamprocess found in the 24h before {acc_time.cern_utc_string()}"
        )

    exact_time = acc_time.__class__.from_timestamp(event_ts[-1])
    LOG.debug(f"Exact time for beamprocess found: {exact_time.cern_utc_string()}")
    return exact_time
