from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING

from pylhc.nxcal_knobs import NXCalResult, get_knob_vals

# Load corrector knobs
if TYPE_CHECKING:
    from pyspark.sql import SparkSession

CURRENT_DIR = Path(__file__).resolve().parent


def get_mcb_vals(spark: SparkSession, time: datetime, beam: int) -> list[NXCalResult]:
    """
    Get MCB K-values for a specific time and beam.

    Args:
        spark (SparkSession): The Spark session.
        time (datetime): The time for which to get the MCB values.
        beam (int): The beam number (1 or 2).

    Returns:
        List of NXCalResult with knob values.
    """
    corrector_knobs = set()
    with CURRENT_DIR.joinpath("constants/corrector_magnets.txt").open() as f:
        for line in f:
            corrector_knobs.add(line.strip())

    beam_pattern = f"%RCB%B{beam}:I_MEAS"
    both_pattern = "RPMBB%RCBX%:I_MEAS"
    patterns = [beam_pattern, both_pattern]

    return get_knob_vals(spark, time, beam, patterns, corrector_knobs, "MCB: ")


if __name__ == "__main__":
    from zoneinfo import ZoneInfo

    from nxcals.spark_session_builder import get_or_create

    logging.basicConfig(level=logging.INFO)

    spark = get_or_create()

    # Input datetime strings
    start_str = "2025-04-19 21:40"
    tz = ZoneInfo("Europe/Zurich")

    # Convert (encode) strings into datetime objects
    start_dt = datetime.strptime(start_str, "%Y-%m-%d %H:%M").replace(tzinfo=tz)

    # Example: show results
    print("Start:", start_dt)
    mcb_vals = get_mcb_vals(spark, start_dt, 1)
    for val in mcb_vals:
        print(val)
    # print(get_mcb_vals(spark, start_dt, 2))
