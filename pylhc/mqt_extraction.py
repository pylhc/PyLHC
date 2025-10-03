import logging
from datetime import datetime

# import jpype
from pyspark.sql import SparkSession

from pylhc.nxcal_knobs import NXCalResult, get_energy, get_knob_vals

logger = logging.getLogger(__name__)


def get_mqts(beam: int) -> set[str]:
    """
    Generate the set of MAD-X MQT (Quadrupole Trim) variable names for a given beam.

    Args:
        beam (int): The beam number (1 or 2).

    Returns:
        set[str]: A set of MAD-X variable names for MQT magnets, e.g., 'kqt12.a12b1'.

    Raises:
        ValueError: If beam is not 1 or 2.

    Examples:
        >>> get_mqts(1)
        {'kqt12.a12b1', 'kqt12.a23b1', ..., 'kqtd.a81b1'}
    """
    if beam not in (1, 2):
        raise ValueError("Beam must be 1 or 2")

    types = ["f", "d"]
    arcs = [12, 23, 34, 45, 56, 67, 78, 81]
    return {f"kqt{t}.a{a}b{beam}" for t in types for a in arcs}


def get_mqt_vals(spark: SparkSession, time: datetime, beam: int) -> list[NXCalResult]:
    """
    Retrieve MQT (Quadrupole Trim) knob values from NXCALS for a specific time and beam.

    This function queries NXCALS for current measurements of MQT power converters,
    calculates the corresponding K-values (integrated quadrupole strengths) using LSA,
    and returns them in MAD-X format with timestamps.

    Args:
        spark (SparkSession): Active Spark session for NXCALS queries.
        time (datetime): The timestamp for which to retrieve the data (timezone-aware recommended).
        beam (int): The beam number (1 or 2).

    Returns:
        list[NXCalResult]: List of NXCalResult objects containing the MAD-X knob names, K-values, and timestamps.

    Raises:
        ValueError: If beam is not 1 or 2 (propagated from get_mqts).
        RuntimeError: If no data is found in NXCALS or LSA calculations fail.
    """
    madx_mqts = get_mqts(beam)
    pattern = f"RPMBB.UA%.RQT%.A%B{beam}:I_MEAS"
    patterns = [pattern]
    return get_knob_vals(spark, time, beam, patterns, madx_mqts, "MQT: ")


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

    energy, time = get_energy(spark, start_dt)
    print(energy)

    mqt_vals = get_mqt_vals(spark, start_dt, 1)
    print("MQT values:")
    print(mqt_vals)
