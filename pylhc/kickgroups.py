"""
Kickgroups
----------

Functions to list KickGroups and show their Kicks.

.. code-block:: none

    usage: kickgroups.py [-h] {list,info} ...

    KickGroups Functions

    optional arguments:
      -h, --help            show this help message and exit

    Functionality:
      {list,info}
        list                List all KickGroups
        info               Show the info of a given KickGroup


Function ``list``:

.. code-block:: none

    usage: kickgroups.py list [-h] [--root ROOT]
                              [--sort {TIMESTAMP,KICKGROUP}]

    List KickGroups

    optional arguments:
      -h, --help            show this help message and exit
      --root ROOT           KickGroups Root-Directory
      --sort {TIMESTAMP,KICKGROUP}
                            Sort KickGroups


Function ``info``:

.. code-block:: none

    usage: kickgroups.py info [-h] [--root ROOT] [--files FILES] group

    KickGroup Info

    positional arguments:
      group                 KickGroup name

    optional arguments:
      -h, --help            show this help message and exit
      --root ROOT           KickGroups Root-Directory
      --files FILES, -f FILES
                            Optional integer. If a value is given, only show the path
                            to *files* SDDS files from the group. Use negative values
                            to show the last files (most recent kicks), positive
                            values for the first ones. A value of zero means showing
                            all files in the group.
"""
import argparse
import json

from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from dateutil import tz
from omc3.utils import logging_tools
from pandas import DataFrame
from tfs import TfsDataFrame

# fmt: off
from pylhc.constants.kickgroups import (
    AMPX,
    AMPY,
    AMPZ,
    BEAM,
    BEAMPROCESS,
    BUNCH,
    COLUMNS_TO_HEADERS,
    DRIVEN_TUNEX,
    DRIVEN_TUNEY,
    DRIVEN_TUNEZ,
    FILL,
    JSON_FILE,
    KICK_COLUMNS,
    KICK_GROUP_COLUMNS,
    KICKGROUP,
    KICKGROUPS_ROOT,
    LOCALTIME,
    OPTICS,
    OPTICS_URI,
    SDDS,
    TIMESTAMP,
    TUNEX,
    TUNEY,
    TURNS,
    UTCTIME,
)

# fmt: on

LOG = logging_tools.get_logger(__name__)

# List Kickgroups --------------------------------------------------------------


def list_available_kickgroups(by: str = TIMESTAMP, root: Path | str = KICKGROUPS_ROOT, printout: bool = True) -> DataFrame:
    """
    List all available KickGroups in `root` with optional sorting..

    Args:
        by (str): Column to sort the KickGroups by. Should be either the
            ``TIMESTAMP`` or ``KICKGROUP`` variable.
        root (pathlib.Path): Alternative `~pathlib.Path` to the KickGroup folder. (Defaults
            to the ``NFS`` path of our kickgroups).
        printout (bool): whether to print out the dataframe, defaults to `True`.

    Returns:
        A `~pandas.DataFrame` with the KickGroups loaded, sorted by the provided
        *by* parameter.
    """
    LOG.debug(f"Listing KickGroups in '{Path(root).absolute()}'")
    kickgroup_paths = get_folder_json_files(root)
    df_info = DataFrame(index=range(len(kickgroup_paths)), columns=KICK_GROUP_COLUMNS)
    for idx, kick_group in enumerate(kickgroup_paths):
        LOG.debug(f"Loading kickgroup info from '{kick_group.absolute()}'")
        data = _load_json(kick_group)
        df_info.loc[idx, KICKGROUP] = data["groupName"]
        df_info.loc[idx, TIMESTAMP] = data["groupCreationTime"]
        df_info.loc[idx, UTCTIME] = _ts_to_datetime(df_info.loc[idx, TIMESTAMP])
        df_info.loc[idx, LOCALTIME] = _utc_to_local(df_info.loc[idx, UTCTIME])
    df_info = df_info.sort_values(by=by).set_index(TIMESTAMP)

    if printout:
        LOG.debug("Here is information about the loaded KickGroups")
        print(df_info.to_string(index=False, formatters=_time_formatters(), justify="center"))

    return df_info


def get_folder_json_files(root: Path | str = KICKGROUPS_ROOT) -> list[Path]:
    """Returns a list of all **.json** files in the folder.

    Args:
        root (Path | str)): the path to the folder. (Defaults
            to the ``NFS`` path of our kickgroups).

    Returns:
        A `list` of `~pathlib.Path` objects to all **json** files within the provided
        *root* parameter.
    """
    LOG.debug(f"Globing for JSON files in {Path(root).absolute()}''")
    return list(Path(root).glob("*.json"))


# Kickgroup Info ---------------------------------------------------------------


def get_kickgroup_info(kick_group: str, root: Path | str = KICKGROUPS_ROOT) -> TfsDataFrame:
    """
    Gather all important info about the KickGroup into a `~tfs.TfsDataFrame`.

    Args:
        kick_group (str): the KickGroup name, corresponds to the kickgroup file name without
            the ``.json`` extension.
        root (pathlib.Path): Alternative `~pathlib.Path` to the KickGroup folder. (Defaults
            to the ``NFS`` path of our kickgroups).

    Returns:
        A `~tfs.TfsDataFrame` with the KickGroup information loaded.
    """
    LOG.debug(f"Loading info from all KickFiles in KickGroup '{kick_group}'")
    kick_group_data = _load_json(Path(root) / f"{kick_group}.json")
    kicks_files = kick_group_data["jsonFiles"]
    df_info = TfsDataFrame(index=range(len(kicks_files)), columns=KICK_COLUMNS, headers={KICKGROUP: kick_group})

    if not len(kicks_files):
        raise ValueError(f"KickGroup {kick_group} contains no kicks.")

    for idx, kf in enumerate(kicks_files):
        df_info.loc[idx, :] = load_kickfile(kf)

    for column in COLUMNS_TO_HEADERS:
        df_info.headers[column] = df_info[column][0]

    return df_info


def load_kickfile(kickfile: Path | str) -> pd.Series:
    """
    Load the important data from a **json** kickfile into a `~pandas.Series`.

    Args:
        kickfile (Path | str): the path to the kickfile to load data from.

    Returns:
        A `~pandas.Series` with the relevant information loaded from the provided
        *kickfile*. The various entries in the Series are defined in `pylhc.constants.kickgroups`
        as ``KICK_COLUMNS``.
    """
    LOG.debug(f"Loading kick information from Kickfile at '{Path(kickfile).absolute()}'")
    kickfile = _find_existing_file_path(kickfile)
    kick = _load_json(kickfile)

    data = pd.Series(index=KICK_COLUMNS, dtype=object)
    data[JSON_FILE] = kickfile
    data[LOCALTIME] = _jsontime_to_datetime(kick["acquisitionTime"])
    data[UTCTIME] = _local_to_utc(data[LOCALTIME])

    try:
        data[SDDS] = _find_existing_file_path(kick["sddsFile"])
    except FileNotFoundError as e:
        LOG.warning(str(e))
        data[SDDS] = None

    data[FILL] = _get_fill_from_path(kick["sddsFile"])  # TODO: Ask OP to include in json?
    data[BEAM] = kick["measurementEnvironment"]["lhcBeam"]["beamName"]
    data[BEAMPROCESS] = kick["measurementEnvironment"]["environmentContext"]["name"]
    data[TURNS] = kick["acqSettings"]["capturedTurns"]
    data[BUNCH] = kick["acqSettings"]["bunchSelection"]
    data[OPTICS] = kick["measurementEnvironment"]["opticsModel"]["opticName"]
    data[OPTICS_URI] = kick["measurementEnvironment"]["opticsModel"]["opticModelURI"]

    three_d = "3D" in kick["excitationSettings"][0]["type"]

    if three_d:
        LOG.debug("Kick is 3D Excitation, loading longitudinal kick settings")
        idx = _get_plane_index(kick["excitationSettings"][0]["acDipoleSettings"], "X")
        idy = _get_plane_index(kick["excitationSettings"][0]["acDipoleSettings"], "Y")

        data[TUNEX] = kick["excitationSettings"][0]["acDipoleSettings"][idx]["measuredTune"]
        data[TUNEY] = kick["excitationSettings"][0]["acDipoleSettings"][idy]["measuredTune"]
        data[DRIVEN_TUNEX] = data[TUNEX] + kick["excitationSettings"][0]["acDipoleSettings"][idx]["deltaTuneStart"]
        data[DRIVEN_TUNEY] = data[TUNEY] + kick["excitationSettings"][0]["acDipoleSettings"][idy]["deltaTuneStart"]
        data[DRIVEN_TUNEZ] = kick["excitationData"][0]["rfdata"]["excitationFrequency"]
        data[AMPX] = kick["excitationSettings"][0]["acDipoleSettings"][idx]["amplitude"]
        data[AMPY] = kick["excitationSettings"][0]["acDipoleSettings"][idy]["amplitude"]
        data[AMPZ] = kick["excitationSettings"][0]["longitudinalRfSettings"]["excitationAmplitude"]
    else:
        LOG.debug("Kick is 2D Excitation, longitudinal settings will be set as NaNs")
        entry_map = {"X": (TUNEX, DRIVEN_TUNEX, AMPX), "Y": (TUNEY, DRIVEN_TUNEY, AMPY)}
        for plane in ["X", "Y"]:
            tune, driven_tune, amp = entry_map[plane]

            data[tune] = np.NaN
            data[driven_tune] = np.NaN
            data[amp] = np.NaN

            try:
                idx = _get_plane_index(kick["excitationSettings"], plane)
            except ValueError as e:
                LOG.warning(f"{str(e)} in {kickfile}")
                continue

            if "measuredTune" not in kick["excitationSettings"][idx]:  # Happens in very early files in 2022
                LOG.warning(f"No measured tune {plane} in the kick file: {kickfile}")
                continue

            data[tune] = kick["excitationSettings"][idx]["measuredTune"]
            data[driven_tune] = data[tune] + _get_delta_tune(kick, idx)
            data[amp] = kick["excitationSettings"][idx]["amplitude"]

        data[DRIVEN_TUNEZ] = np.NaN
        data[AMPZ] = np.NaN

    return data

def _get_delta_tune(kick: dict, idx_plane: int) -> float:
    """ Return the delta from the tune for the kicks.
    For some reason, there are multiple different keys where this can be stored. """

    # Default key for ACDipole ---
    # There is also "deltaTuneEnd", but we usually don't change the delta during kick
    try:
        return kick["excitationSettings"][idx_plane]["deltaTuneStart"]
    except KeyError:
        pass

    # Key for ADTACDipole ---
    try:
        return kick["excitationSettings"][idx_plane]["deltaTune"]
    except KeyError:
        pass

    # Another key for ADTACDipole (unclear to me why) ---
    try:
        return kick["excitationSettings"][idx_plane]["deltaTuneOffset"]
    except KeyError:
        pass

    raise KeyError(f"Could not find delta tune for plane-entry {idx_plane}")


def _find_existing_file_path(path: str|Path) -> Path:
    """ Find the existing kick file for the kick group. """
    path = Path(path)
    if path.is_file():
        return path

    fill_data = "FILL_DATA"
    all_fill_data = "ALL_FILL_DATA"

    if fill_data in path.parts:
        # Fills are moved at the end of year
        idx = path.parts.index(fill_data)+1
        new_path = Path(*path.parts[:idx], all_fill_data, *path.parts[idx:])
        if new_path.exists():
            return new_path

    raise FileNotFoundError(f"Could not find kick file at {path}")



# Functions with console output ---

# Full Info -


def show_kickgroup_info(kick_group: str, root: Path | str = KICKGROUPS_ROOT) -> None:
    """
    Wrapper around `~pylhc.kickgroups.get_kickgroup_info`, gathering the relevant
    information from the kick files in the group and printing it to console.

    Args:
        kick_group (str): the KickGroup name, corresponds to the kickgroup file name without
            the ``.json`` extension.
        root (pathlib.Path): Alternative `~pathlib.Path` to the KickGroup folder. (Defaults
            to the ``NFS`` path of our kickgroups).
    """
    kicks_info = get_kickgroup_info(kick_group, root)
    _print_kickgroup_info(kicks_info)


def _print_kickgroup_info(kicks_info: TfsDataFrame) -> None:
    """
    Print the full info about the kickgroup.

    Args:
        kicks_info (TfsDataFrame): Gathered Kickgroup data.
    """
    for header, value in kicks_info.headers.items():
        print(f"{header}: {value}")
    print()
    print(
        kicks_info.drop(columns=COLUMNS_TO_HEADERS).to_string(
            index=False, na_rep=" - ", justify="center", formatters=_time_formatters()
        )
    )


# Files only -


def show_kickgroup_files(kick_group: str, nfiles: int = None, root: Path | str = KICKGROUPS_ROOT) -> None:
    """
    Wrapper around `pylhc.kickgroups.get_kickgroup_info`, gathering the relevant
    information from all kickfiles in the KickGroup and printing only the sdds-filepaths
    to console.

    Args:
        kick_group (str): the KickGroup name, corresponds to the kickgroup file name without
            the ``.json`` extension.
        nfiles (int): Number of files to show. Use negative values for the last nfiles.
                      A value of zero or None means all files in the group.
        root (pathlib.Path): Alternative `~pathlib.Path` to the KickGroup folder. (Defaults
            to the ``NFS`` path of our kickgroups).
    """
    kicks_info = get_kickgroup_info(kick_group, root)
    _print_kickgroup_files(kicks_info, nfiles=nfiles)


def _print_kickgroup_files(kicks_info: TfsDataFrame, nfiles: int = None) -> None:
    """
    Print *nfiles* from the KickGroup as space-separated quoted strings, which can
    then be directly copy-pasted into the GUI to load them at once.

    Args:
        kicks_info (TfsDataFrame): A `~tfs.TfsDataFrame` with the gathered KickGroup data.
        nfiles (int): Number of files to show. Use negative values for the last nfiles.
            A value of zero or `None` means all files in the group.
    """
    kickgroup = kicks_info.headers[KICKGROUP]
    nfiles_total = len(kicks_info.index)

    nfiles_str = "all files"
    if nfiles is None or nfiles == 0:
        element_slice = slice(None, None)
    else:
        nfiles_str = f"{'last ' if nfiles < 0 else ''}{abs(nfiles)} file(s)"
        if abs(nfiles) > nfiles_total:
            LOG.warning(
                f"You requested a total of {abs(nfiles)} files to print"
                f" but there are only {nfiles_total} kicks in {kickgroup}."
            )
            nfiles = nfiles_total
        element_slice = slice(nfiles) if nfiles > 0 else slice(nfiles, None)

    print(f"Kickgroup {kicks_info.headers[KICKGROUP]}, {nfiles_str}:")
    print(" ".join([f'"{s}"' for s in kicks_info[SDDS][element_slice]]))


# Helper -----------------------------------------------------------------------

# IO ---


def _load_json(jsonfile: Path | str) -> dict:
    return json.loads(Path(jsonfile).read_text())


# Time ---


def _ts_to_datetime(ts: int) -> datetime:
    return datetime.utcfromtimestamp(ts / 1000)


def _jsontime_to_datetime(time_str: str) -> datetime:
    return datetime.strptime(time_str, "%d-%m-%y_%H-%M-%S")


def _datetime_to_string(dt: datetime):
    return dt.strftime("  %Y-%m-%d %H:%M:%S")


def _time_formatters():
    return {UTCTIME: _datetime_to_string, LOCALTIME: _datetime_to_string}


def _utc_to_local(dt: datetime):
    return dt.replace(tzinfo=tz.gettz("UTC")).astimezone(tz.gettz("Europe/Paris"))


def _local_to_utc(dt: datetime):
    return dt.replace(tzinfo=tz.gettz("Europe/Paris")).astimezone(tz.gettz("UTC"))


# Other ---


def _get_plane_index(data: list[dict], plane: str) -> str:
    """
    Find the index for the given plane in the data list.
    This is necessary as they are not always in X,Y order.
    """
    name = {"X": "HORIZONTAL", "Y": "VERTICAL"}[plane]
    for idx, entry in enumerate(data):
        if entry["plane"] == name:
            return idx
    else:
        raise ValueError(f"Plane '{plane}' not found in data.")


def _get_fill_from_path(sdds_path: str | Path) -> str:
    """ Get the fill number from the path to the sdds file.
    Note: Not sure why the fill is not saved automatically into the .json file.
    Maybe we should ask OP to include this.
    """
    parts = Path(sdds_path).parts
    idx_parent = parts.index("FILL_DATA")
    return int(parts[idx_parent + 1])


# Script Mode ------------------------------------------------------------------


def _get_args():
    """Parse Commandline Arguments."""
    # argparse is a bit weird: you need to create the normal parser
    # AND a parent parser if you want to use the same arg in both subparsers.
    # Using the main parser also as parent will result in `function` being
    # always `None`.
    parser = argparse.ArgumentParser(description="KickGroups Functions")
    parent_parser = argparse.ArgumentParser()
    parent_parser.add_argument(
        "--root",
        type=str,
        required=False,
        default=KICKGROUPS_ROOT,
        dest="root",
        help="KickGroups Root-Directory",
    )
    subparsers = parser.add_subparsers(
        title="Functionality",
        dest="function",
        required=True,
    )
    # ----- Full KickGroup Parser ----- #
    parser_kickgroups = subparsers.add_parser(
        "list",
        parents=[parent_parser],
        add_help=False,
        description="List KickGroups",
        help="List all KickGroups",
    )
    parser_kickgroups.add_argument(
        "--sort",
        type=str,
        dest="sort",
        help="Sort KickGroups",
        choices=[TIMESTAMP, KICKGROUP],
        default=TIMESTAMP,
    )
    # ---- KickGroup Info Parser ---- #
    parser_info = subparsers.add_parser(
        "info",
        parents=[parent_parser],
        add_help=False,
        description="KickGroup Info",
        help="Show the info of a given KickGroup",
    )
    parser_info.add_argument(
        "group",
        type=str,
        help="KickGroup name",
    )
    parser_info.add_argument(
        "--files",
        "-f",
        dest="files",
        type=int,
        nargs="?",
        help="Optional integer. If a value is given, only show the path to *files* SDDS files from the group. "
        "Use negative values to show the last files (most recent kicks), positive values for the first ones. "
        "A value of zero means showing all files in the group.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    options = _get_args()
    if options.function == "list":
        list_available_kickgroups(by=options.sort, root=options.root)

    if options.function == "info":
        if options.files is None:
            show_kickgroup_info(kick_group=options.group, root=options.root)
        else:
            show_kickgroup_files(kick_group=options.group, root=options.root, nfiles=options.files)
