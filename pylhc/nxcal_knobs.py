import logging
from dataclasses import dataclass
from datetime import datetime, timedelta

# import jpype
import pandas as pd
from nxcals.api.extraction.data.builders import DataQuery
from pyspark.sql import SparkSession
from pyspark.sql import functions as f
from pyspark.sql.window import Window

from pylhc.data_extract.lsa import LSAClient

logger = logging.getLogger(__name__)


@dataclass
class NXCalResult:
    name: str
    value: float
    timestamp: pd.Timestamp


def get_raw_vars(spark: SparkSession, time: datetime, var_name: str) -> list[NXCalResult]:
    """
    Retrieve raw variable values from NXCALS.

    Parameters
    ----------
    spark : SparkSession
        Active Spark session
    time : datetime
        Python datetime (timezone-aware recommended)
    var_name : str
        Name or pattern of the variable(s) to retrieve

    Returns
    -------
    dict[str, tuple[float, pd.Timestamp, str]]
       Dictionary with full variable name as key, and (value, timestamp, full variable name) as value
       for the latest sample of each matching variable at the given time
    """

    # Look back 1 day to be sure we get at least one sample
    start_time = time - timedelta(days=1)
    end_time = time
    logger.info(f"Retrieving raw variables {var_name} from {start_time} to {end_time}")

    df = (
        DataQuery.builder(spark)
        .variables()
        .system("CMW")
        .nameLike(var_name)
        .timeWindow(start_time, end_time)
        .build()
    )

    # Avoid full count() – just check if we have at least one row
    if df is None or not df.take(1):
        raise RuntimeError(f"No data found for {var_name} in the given interval.")

    logger.info(f"Raw variables {var_name} retrieved successfully.")

    # Get the latest sample for each variable
    window_spec = Window.partitionBy("nxcals_variable_name").orderBy(
        f.col("nxcals_timestamp").desc()
    )
    latest_df = (
        df.withColumn("row_num", f.row_number().over(window_spec))
        .filter(f.col("row_num") == 1)
        .select("nxcals_variable_name", "nxcals_value", "nxcals_timestamp")
    )

    results = []
    for row in latest_df.collect():
        full_varname = row["nxcals_variable_name"]
        raw_val = float(row["nxcals_value"])
        ts = pd.to_datetime(row["nxcals_timestamp"], unit="ns", utc=True)
        results.append(NXCalResult(full_varname, raw_val, ts))
        logger.info(f"LHC value retrieved: {full_varname} = {raw_val:.2f} at {ts}")

    return results


def get_energy(spark: SparkSession, time: datetime) -> tuple[float, pd.Timestamp]:
    """

    Args:
        spark (SparkSession): Active Spark session
        time (datetime): Python datetime (timezone-aware recommended)

    Returns:
        (energy_gev, timestamp): Beam energy in GeV and its timestamp
    """
    scale = 0.120
    raw_vars = get_raw_vars(spark, time, "HX:ENG")
    if not raw_vars:
        raise RuntimeError("No energy data found.")
    return raw_vars[0].value * scale, raw_vars[0].timestamp


def map_pc_name_to_madx(pc_name: str) -> str:
    """
    Convert a power converter name or circuit name to its corresponding MAD-X name.

    This function processes the input name by removing the ':I_MEAS' suffix if present,
    extracting the circuit name from full power converter names (starting with 'RPMBB' or 'RPL'),
    applying specific string replacements for MAD-X compatibility, and converting the result to lowercase.

    Args:
        pc_name (str): The power converter or circuit name to convert.

    Returns:
        str: The converted MAD-X name.

    Examples:
        >>> map_pc_name_to_madx("RPMBB.UJ33.RCBH10.L1B1:I_MEAS")
        'acb10.l1b1'
        >>> map_pc_name_to_madx("RQT12.L5B1")
        'kqt12.l5b1'
    """
    # Remove the ':I_MEAS' suffix if it exists
    pc_name = strip_i_meas(pc_name)

    # Extract circuit name from full power converter names
    if "." in pc_name:
        parts = pc_name.split(".")
        # For full names like 'RPMBB.UJ33.RCBH10.L1B1', take from the third part onward, else use pc_name
        circuit_name = ".".join(parts[2:]) if len(parts) > 2 else pc_name
    else:
        circuit_name = pc_name

    # Apply MAD-X specific replacements
    replacements = {"RQT": "KQT", "RCB": "ACB"}
    for old, new in replacements.items():
        circuit_name = circuit_name.replace(old, new)

    # Return in lowercase as required by MAD-X
    return circuit_name.lower()


def strip_i_meas(text: str) -> str:
    """
    Remove the I_MEAS suffix from a variable name.
    """
    return text.removesuffix(":I_MEAS")


def get_knob_vals(
    spark: SparkSession,
    time: datetime,
    beam: int,
    patterns: list[str],
    expected_knobs: set[str] | None = None,
    log_prefix: str = "",
) -> list[NXCalResult]:
    """
    Retrieve knob values for a given beam and time using specified patterns.

    Args:
        spark: Spark session
        time: Time to retrieve data for
        beam: Beam number
        patterns: List of variable patterns to retrieve
        expected_knobs: Set of expected MAD-X knob names. If None, returns all found knobs.
        log_prefix: Prefix for logging messages

    Returns:
        List of NXCalResult with knob values
    """
    logger.info(f"{log_prefix}Starting data retrieval for beam {beam} at time {time}")

    # Retrieve raw current measurements from NXCALS
    combined_vars: list[NXCalResult] = []
    for pattern in patterns:
        logger.info(f"{log_prefix}Getting currents for pattern {pattern} at {time}")
        raw_vars = get_raw_vars(spark, time, pattern)
        combined_vars.extend(raw_vars)

    # Get beam energy for K-value calculations
    energy, _ = get_energy(spark, time)

    # Prepare currents dict for LSA
    currents = {strip_i_meas(var.name): var.value for var in combined_vars}

    logger.info(
        f"{log_prefix}Calculating K values for {len(currents)} power converters with energy {energy} GeV"
    )

    # Calculate K values using LSA
    lsa_client = LSAClient()
    k_values = lsa_client.calc_k_from_iref(currents, energy)

    # Transform K values keys to MAD-X format
    k_values_madx = {map_pc_name_to_madx(key): value for key, value in k_values.items()}

    found_keys = set(k_values_madx.keys())

    if expected_knobs is None:
        expected_knobs = found_keys

    if missing_keys := expected_knobs - found_keys:
        logger.warning(f"{log_prefix}Missing K-values for knobs: {missing_keys}")
        print(len(missing_keys))

    if unknown_keys := found_keys - expected_knobs:
        logger.warning(f"{log_prefix}Unknown K-values found: {unknown_keys}")
        print(len(unknown_keys))

    # Build result list with timestamps
    timestamps = {map_pc_name_to_madx(var.name): var.timestamp for var in combined_vars}
    results = []
    for madx_name in expected_knobs:
        value = k_values_madx.get(madx_name)
        timestamp = timestamps.get(madx_name)
        if value is not None and timestamp is not None:
            results.append(NXCalResult(madx_name, value, timestamp))
        else:
            logger.warning(f"{log_prefix}Missing data for {madx_name}")

    return results
