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

    usage: kickgroups.py info [-h] [--root ROOT] group

    KickGroup Info

    positional arguments:
      group         KickGroup name

    optional arguments:
      -h, --help   show this help message and exit
      --root ROOT  KickGroups Root-Directory

"""
import argparse
import json

from datetime import datetime
from pathlib import Path
from typing import List, Union

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


def list_available_kickgroups(by: str = TIMESTAMP, root: Union[Path, str] = KICKGROUPS_ROOT, printout: bool = True) -> DataFrame:
    """
    List all available KickGroups in `root` with optional sorting..

    Args:
        by (str): Column to sort the KickGroups by. Should be either the
            ``TIMESTAMP`` or ``KICKGROUP`` variable.
        root (Path): Alternative `~pathlib.Path` to the KickGroup folder. (Defaults
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
        LOG.debug(f"Here is information about the loaded KickGroups")
        print(df_info.to_string(index=False, formatters=_time_formatters(), justify="center"))

    return df_info


def get_folder_json_files(root: Union[Path, str] = KICKGROUPS_ROOT) -> List[Path]:
    """Returns a list of all **.json** files in the folder.

    Args:
        root (Union[Path, str])): the path to the folder. (Defaults
            to the ``NFS`` path of our kickgroups).

    Returns:
        A `list` of `~pathlib.Path` objects to all **json** files within the provided
        *root* parameter.
    """
    LOG.debug(f"Globing for JSON files in {Path(root).absolute()}''")
    return list(Path(root).glob("*.json"))


# Kickgroup Info ---------------------------------------------------------------


def kickgroup_info(kick_group: str, root: Union[Path, str] = KICKGROUPS_ROOT, printout: bool = True) -> TfsDataFrame:
    """
    Gather all important info about the KickGroup into a `~tfs.TfsDataFrame` and print it.

    Args:
        kick_group (str): the KickGroup name, corresponds to the kickgroup file name without
            the ``.json`` extension.
        root (Path): Alternative `~pathlib.Path` to the KickGroup folder. (Defaults
            to the ``NFS`` path of our kickgroups).
        printout (bool): whether to print out the dataframe, defaults to `True`.

    Returns:
        A `~tfs.TfsDataFrame` with the KickGroup information loaded.
    """
    LOG.debug(f"Loading info from all KickFiles in KickGroup '{kick_group}'")
    kick_group_data = _load_json(Path(root) / f"{kick_group}.json")
    kicks_files = kick_group_data["jsonFiles"]
    df_info = TfsDataFrame(index=range(len(kicks_files)), columns=KICK_COLUMNS, headers={KICKGROUP: kick_group})

    if not len(kicks_files):
        raise FileNotFoundError(f"KickGroup {kick_group} contains no kicks.")

    for idx, kf in enumerate(kicks_files):
        df_info.loc[idx, :] = load_kickfile(kf)

    for column in COLUMNS_TO_HEADERS:
        df_info.headers[column] = df_info[column][0]

    if printout:
        _print_kickgroup_info(df_info)

    return df_info


def load_kickfile(kickfile: Union[Path, str]) -> pd.Series:
    """
    Load the important data from a **json** kickfile into a `~pandas.Series`.

    Args:
        kickfile (Union[Path, str]): the path to the kickfile to load data from.

    Returns:
        A `~pandas.Series` with the relevant information loaded from the provided
        *kickfile*. The various entries in the Series are defined in `pylhc.constants.kickgroups`
        as ``KICK_COLUMNS``.
    """
    LOG.debug(f"Loading kick information from Kickfile at '{Path(kickfile).absolute()}'")
    kick = _load_json(kickfile)
    data = pd.Series(index=KICK_COLUMNS, dtype=object)
    data[LOCALTIME] = _jsontime_to_datetime(kick["acquisitionTime"])
    data[UTCTIME] = _local_to_utc(data[LOCALTIME])
    data[SDDS] = kick["sddsFile"]
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
        LOG.debug(f"Kick is 2D Excitation, longitudinal settings will be set as NaNs")
        idx = _get_plane_index(kick["excitationSettings"], "X")
        idy = _get_plane_index(kick["excitationSettings"], "Y")

        data[TUNEX] = kick["excitationSettings"][idx]["measuredTune"]
        data[TUNEY] = kick["excitationSettings"][idy]["measuredTune"]
        data[DRIVEN_TUNEX] = data[TUNEX] + kick["excitationSettings"][idx]["deltaTuneStart"]
        data[DRIVEN_TUNEY] = data[TUNEY] + kick["excitationSettings"][idy]["deltaTuneStart"]
        data[DRIVEN_TUNEZ] = np.NaN
        data[AMPX] = kick["excitationSettings"][idx]["amplitude"]
        data[AMPY] = kick["excitationSettings"][idy]["amplitude"]
        data[AMPZ] = np.NaN

    return data


def _print_kickgroup_info(kicks_info: TfsDataFrame):
    """Actually print the full info about the kickgroup."""
    for header, value in kicks_info.headers.items():
        print(f"{header}: {value}")
    print()
    print(
        kicks_info.drop(columns=COLUMNS_TO_HEADERS).to_string(
            index=False, na_rep=" - ", justify="center", formatters=_time_formatters()
        )
    )


# Helper -----------------------------------------------------------------------

# IO ---


def _load_json(jsonfile: Union[Path, str]) -> dict:
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

def _get_plane_index(data: List[dict], plane: str):
    """Find the index for the given plane in the data list.
    This is necessary as they are not always in X,Y order.
    """
    name = {'X': 'HORIZONTAL', 'Y': 'VERTICAL'}[plane]
    for idx, entry in enumerate(data):
        if entry['plane'] == name:
            return idx
    else:
        raise ValueError(f"Plane '{plane}' not found in data.")


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
    return parser.parse_args()


if __name__ == "__main__":
    options = _get_args()
    if options.function == "list":
        list_available_kickgroups(by=options.sort, root=options.root)

    if options.function == "info":
        kickgroup_info(kick_group=options.group, root=options.root)
